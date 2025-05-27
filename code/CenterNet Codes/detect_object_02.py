import torch
import numpy as np
import cv2
from torchvision import transforms
import albumentations as A
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

class DentalAugmentation:
    def __init__(self, p=0.5):
        self.transform = A.Compose([
            # Intensity adjustments to simulate different exposure levels
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=p
            ),
            # Simulate different X-ray angles
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=p
            ),
            # Add noise to simulate different X-ray qualities
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=p
            ),
            # Adjust gamma to simulate different tissue densities
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=p
            ),
            # Elastic deformation to simulate tissue variation
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=cv2.BORDER_CONSTANT,
                p=p
            )
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
    
    def __call__(self, image, boxes=None):
        if boxes is None:
            transformed = self.transform(image=image)
            return transformed['image']
        
        # Prepare boxes for augmentation
        labels = ['tooth'] * len(boxes)
        transformed = self.transform(
            image=image,
            bboxes=boxes,
            labels=labels
        )
        
        return transformed['image'], transformed['bboxes']

def post_process_detections(heatmap, wh, offset, confidence_threshold=0.3, nms_threshold=0.5):
    """
    Process CenterNet outputs to get final bounding box predictions
    """
    batch_size = heatmap.shape[0]
    height = heatmap.shape[2]
    width = heatmap.shape[3]
    num_classes = heatmap.shape[1]
    
    # Get peaks in the heatmap
    heatmap = torch.nn.functional.max_pool2d(
        heatmap,
        kernel_size=3,
        padding=1,
        stride=1
    )
    
    scores, indices = torch.max(heatmap, dim=1)
    scores = scores.view(batch_size, -1)
    topk_scores, topk_inds = torch.topk(scores, k=100)
    
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    
    # Get width and height predictions
    wh = wh.view(batch_size, 2, -1)
    offset = offset.view(batch_size, 2, -1)
    
    topk_wh = torch.gather(wh, 2, 
        topk_inds.unsqueeze(1).expand(batch_size, 2, 100))
    topk_offset = torch.gather(offset, 2,
        topk_inds.unsqueeze(1).expand(batch_size, 2, 100))
    
    # Calculate bounding box coordinates
    boxes = torch.zeros_like(topk_wh)
    boxes[:, 0] = topk_xs + topk_offset[:, 0]  # x1
    boxes[:, 1] = topk_ys + topk_offset[:, 1]  # y1
    
    # Convert center to corners
    boxes[:, 0] -= topk_wh[:, 0] / 2  # x1
    boxes[:, 1] -= topk_wh[:, 1] / 2  # y1
    boxes[:, 0] += topk_wh[:, 0]      # x2
    boxes[:, 1] += topk_wh[:, 1]      # y2
    
    # Filter by confidence
    mask = topk_scores > confidence_threshold
    boxes = boxes[mask]
    scores = topk_scores[mask]
    
    # Apply NMS
    keep = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    
    return boxes, scores

def evaluate_model(model, val_loader, device):
    """
    Evaluate model performance using mean Average Precision (mAP)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            target_heatmaps = batch['heatmap']
            
            # Get predictions
            heatmaps, wh, offset = model(images)
            boxes, scores = post_process_detections(
                heatmaps, wh, offset
            )
            
            # Convert predictions and targets to common format
            all_predictions.extend(scores.cpu().numpy())
            all_targets.extend(target_heatmaps.view(-1).cpu().numpy())
    
    # Calculate mAP
    mAP = average_precision_score(all_targets, all_predictions)
    return mAP

def visualize_predictions(image, boxes, scores, threshold=0.5):
    """
    Visualize detection results on the image
    """
    image = image.copy()
    
    # Filter predictions by confidence
    mask = scores > threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    # Draw boxes
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.int()
        cv2.rectangle(
            image,
            (x1, y1), (x2, y2),
            color=(0, 255, 0),
            thickness=2
        )
        cv2.putText(
            image,
            f'{score:.2f}',
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    return image

def train_with_validation(model, train_loader, val_loader, 
                         optimizer, num_epochs, device):
    """
    Training loop with validation
    """
    best_map = 0
    train_losses = []
    val_maps = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            images = batch['image'].to(device)
            heatmap_target = batch['heatmap'].to(device)
            wh_target = batch['wh'].to(device)
            offset_target = batch['offset'].to(device)
            reg_mask = batch['reg_mask'].to(device)
            
            optimizer.zero_grad()
            
            heatmap_pred, wh_pred, offset_pred = model(images)
            
            # Calculate losses
            loss = calculate_total_loss(
                heatmap_pred, heatmap_target,
                wh_pred, wh_target,
                offset_pred, offset_target,
                reg_mask
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Validation
        mAP = evaluate_model(model, val_loader, device)
        
        # Save best model
        if mAP > best_map:
            best_map = mAP
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP,
            }, 'best_model.pth')
        
        # Record metrics
        train_losses.append(np.mean(epoch_losses))
        val_maps.append(mAP)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_losses[-1]:.4f}')
        print(f'Validation mAP: {mAP:.4f}')
        
        # Plot metrics
        plot_metrics(train_losses, val_maps)

def plot_metrics(train_losses, val_maps):
    """
    Plot training metrics
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot validation mAP
    plt.subplot(1, 2, 2)
    plt.plot(val_maps)
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def calculate_total_loss(heatmap_pred, heatmap_target,
                        wh_pred, wh_target,
                        offset_pred, offset_target,
                        reg_mask):
    """
    Calculate combined loss for all predictions
    """
    heatmap_loss = focal_loss(heatmap_pred, heatmap_target)
    wh_loss = F.l1_loss(
        wh_pred * reg_mask.unsqueeze(1),
        wh_target * reg_mask.unsqueeze(1),
        reduction='sum'
    ) / (reg_mask.sum() + 1e-4)
    offset_loss = F.l1_loss(
        offset_pred * reg_mask.unsqueeze(1),
        offset_target * reg_mask.unsqueeze(1),
        reduction='sum'
    ) / (reg_mask.sum() + 1e-4)
    
    return heatmap_loss + wh_loss + offset_loss