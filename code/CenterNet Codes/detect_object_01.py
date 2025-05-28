import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np

class CenterNetDental(nn.Module):
    def __init__(self, num_classes=1, output_size=(128, 128)):
        super(CenterNetDental, self).__init__()
        
        # ResNet50 backbone with pretrained weights
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Upsampling layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Head layers for different tasks
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        
        self.output_size = output_size

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Upsampling
        features = self.deconv_layers(features)
        
        # Get predictions
        heatmaps = self.heatmap_head(features)
        wh = self.wh_head(features)
        offset = self.offset_head(features)
        
        # Apply sigmoid to heatmaps
        heatmaps = torch.sigmoid(heatmaps)
        
        return heatmaps, wh, offset

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    
    return heatmap

class DentalDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, annotations, output_size=(128, 128), transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.output_size = output_size
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = load_image(self.image_paths[idx])  # Implement load_image based on your needs
        
        if self.transform:
            img = self.transform(img)
        
        # Get annotations for this image
        boxes = self.annotations[idx]
        
        # Create target maps
        heatmap = np.zeros((1, *self.output_size))
        wh = np.zeros((2, *self.output_size))
        offset = np.zeros((2, *self.output_size))
        reg_mask = np.zeros((*self.output_size))
        
        # Process each box
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            # Scale to output size
            cx = cx * self.output_size[1] / img.shape[1]
            cy = cy * self.output_size[0] / img.shape[0]
            w = w * self.output_size[1] / img.shape[1]
            h = h * self.output_size[0] / img.shape[0]
            
            # Draw gaussian
            draw_gaussian(heatmap[0], (cx, cy), radius=3)
            
            # Add width/height
            wh[0, int(cy), int(cx)] = w
            wh[1, int(cy), int(cx)] = h
            
            # Add offset
            offset[0, int(cy), int(cx)] = cx - int(cx)
            offset[1, int(cy), int(cx)] = cy - int(cy)
            
            # Add reg mask
            reg_mask[int(cy), int(cx)] = 1
            
        return {
            'image': torch.FloatTensor(img),
            'heatmap': torch.FloatTensor(heatmap),
            'wh': torch.FloatTensor(wh),
            'offset': torch.FloatTensor(offset),
            'reg_mask': torch.FloatTensor(reg_mask)
        }

def focal_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    neg_weights = torch.pow(1 - gt, 4)
    
    loss = 0
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
        
    return loss

def train_model(model, train_loader, optimizer, epoch):
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        heatmap_target = batch['heatmap']
        wh_target = batch['wh']
        offset_target = batch['offset']
        reg_mask = batch['reg_mask']
        
        optimizer.zero_grad()
        
        heatmap_pred, wh_pred, offset_pred = model(images)
        
        # Calculate losses
        heatmap_loss = focal_loss(heatmap_pred, heatmap_target)
        wh_loss = F.l1_loss(wh_pred * reg_mask.unsqueeze(1), 
                           wh_target * reg_mask.unsqueeze(1),
                           reduction='sum') / (reg_mask.sum() + 1e-4)
        offset_loss = F.l1_loss(offset_pred * reg_mask.unsqueeze(1),
                               offset_target * reg_mask.unsqueeze(1),
                               reduction='sum') / (reg_mask.sum() + 1e-4)
        
        loss = heatmap_loss + wh_loss + offset_loss
        
        loss.backward()
        optimizer.step()