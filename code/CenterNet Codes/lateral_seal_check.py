import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.patches as patches

VISUALIZE_PLOTS = False  # Set to True to enable visualization

def load_annotations(json_file):
    """Load the annotation JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_image(image_path):
    """Load an image from the given path."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error loading image:", image_path)
    return img

def apply_clahe_color(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the L-channel of the LAB representation to enhance the image details.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def scale_points(points, orig_width, orig_height):
    """
    Convert normalized polygon points to pixel coordinates.
    Assumes the normalized points are given in percentages.
    Adjust the conversion if your points are normalized differently.
    """
    scaled = []
    for pt in points:
        # If points are in [x_percent, y_percent] format (0-100)
        x = int(pt[0])
        y = int(pt[1])
        scaled.append([x, y])
    return scaled

def create_mask_for_polygon(img_shape, points):
    """
    Create a binary mask from a list of polygon points.
    The mask will have the same height and width as the input image.
    """
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def create_border_regions(mask, range_pixels):
    """
    Given a binary mask, create inner and outer border regions using
    morphological erosion and dilation.
    
    - inner_border: region lost when eroding the mask by 'range_pixels'
    - outer_border: additional region gained when dilating the mask by 'range_pixels'
    
    Note: 'range_pixels' is an approximation for 2mm in pixels. Adjust as needed.
    """
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=range_pixels)
    dilated = cv2.dilate(mask, kernel, iterations=range_pixels)
    
    inner_border = cv2.subtract(mask, eroded)
    outer_border = cv2.subtract(dilated, mask)
    return inner_border, outer_border

def analyze_intensities(image, mask, inner_border, outer_border):
    """
    Analyze average grayscale intensities in:
      - The filling region (mask)
      - The inner border (edge within the filling)
      - The outer border (just outside the filling)
      
    The analysis is done on a grayscale version of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filling_values = gray[mask == 255]
    inner_values = gray[inner_border == 255]
    outer_values = gray[outer_border == 255]
    
    avg_filling = np.mean(filling_values) if filling_values.size > 0 else 0
    avg_inner = np.mean(inner_values) if inner_values.size > 0 else 0
    avg_outer = np.mean(outer_values) if outer_values.size > 0 else 0
    
    return avg_filling, avg_inner, avg_outer

def overlay_annotation(image, points, color=(0, 255, 0), alpha=0.4):
    """
    Overlay a filled polygon with transparency on the image.
    """
    overlay = image.copy()
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    combined = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return combined

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.4):
    """
    Overlay a binary mask on the image with a specified color and transparency.
    """
    overlay = image.copy()
    overlay[mask == 255] = color
    combined = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return combined

def mark_low_intensity_pixels(image, outer_mask, threshold=200, mark_color=(0, 0, 255), alpha=0.5):
    """
    Mark pixels in the outer edge region (outer_mask) where the grayscale intensity
    is less than the specified threshold.
    
    The pixels are marked with the provided mark_color.
    
    Returns:
        - The image with marked low-intensity pixels
        - The count of low-intensity pixels
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a boolean mask for pixels in outer_mask with intensity less than threshold
    low_intensity_mask = (outer_mask == 255) & (gray < threshold)
    
    # Count the number of low-intensity pixels
    low_intensity_pixel_count = np.sum(low_intensity_mask)
    
    marked = image.copy()
    # Mark these pixels with mark_color
    marked[low_intensity_mask] = mark_color
    combined = cv2.addWeighted(marked, alpha, image, 1 - alpha, 0)
    
    return combined, low_intensity_pixel_count

def compute_intensity_profile(image, mask, inner_pixels=8, outer_pixels=10):
    """
    Compute the average grayscale intensity in 1-pixel-wide bins
    from inner_pixels inside (-ve) to outer_pixels outside (+ve).
    
    Returns:
        - bins: List of distances from the boundary
        - intensity_profile: Mean grayscale intensity per bin
        - bin_masks: Dictionary mapping bin indices to pixel masks
    """
    bins = np.arange(-inner_pixels, outer_pixels + 1, 1)
    intensity_profile = []
    bin_masks = {}  # Store masks for each bin
    kernel = np.ones((3,3), np.uint8)

    for b in bins:
        if b < 0:
            eroded_1 = cv2.erode(mask, kernel, iterations=abs(b))
            border_1 = cv2.subtract(mask, eroded_1)

            eroded_2 = cv2.erode(mask, kernel, iterations=abs(b) + 1)
            border_2 = cv2.subtract(mask, eroded_2)

            border = cv2.subtract(border_2, border_1)
        else:
            dilated_1 = cv2.dilate(mask, kernel, iterations=b)
            border_1 = cv2.subtract(dilated_1, mask)

            dilated_2 = cv2.dilate(mask, kernel, iterations=b + 1)
            border_2 = cv2.subtract(dilated_2, mask)

            border = cv2.subtract(border_2, border_1)

        # Use grayscale for intensity measurement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        values = gray[border == 255]
        
        mean_intensity = np.mean(values) if values.size > 0 else 0
        intensity_profile.append(mean_intensity)
        bin_masks[b] = border  # Store the mask for this bin
    
    return bins, intensity_profile, bin_masks

# Visualization function (edited to show only "lateral_seal" for "Filling")
def visualize(image_path, annotations, title=""):
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(title)

    for ann in annotations:
        shape = ann.get("shape")
        coords = ann.get("annotations", [])
        name = ann.get("name")

        if shape == "polygon" and name == "Filling" and len(coords) >= 4:
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            poly = patches.Polygon(points, closed=True, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(poly)
            ax.text(points[0][0], points[0][1], name, color='red', fontsize=9)
            
            lateral_seal = ann.get("lateral_seal", "")
            if lateral_seal:
                ax.text(points[0][0], points[0][1]+15, f"Lateral Seal: {lateral_seal}", color='blue', fontsize=8)


    plt.axis('off')
    plt.show()


def main():
    # Define file names and parameters
    json_file = "annotated_and_labeled.json"
    image_dir = "annotated_and_labeled" # Directory containing the images
    range_pixels = 10  # Approximate pixel distance for a 2mm range (adjust as needed)
    
    # Load annotation JSON and image
    data = load_annotations(json_file)

    print(f"Loaded {len(data)} entries from {json_file}")

    for entry in data:
        annotations = entry.get("annotation", [])
        has_valid_filling = any(
            ann.get("shape") == "polygon" and 
            ann.get("name") == "Filling" and 
            len(ann.get("annotations", [])) >= 4
            for ann in annotations
        )

        if has_valid_filling:
            img_path = os.path.join(image_dir, entry["image_name"])
            # Print the image name for debugging
            print(f"Visualizing: {entry['image_name']}")
            
            # Check if the image file exists before visualizing
            if os.path.exists(img_path) and VISUALIZE_PLOTS:
                visualize(img_path, annotations, title=entry["image_name"])

        annotations = entry.get('annotation', [])
        has_valid_filling = any(
            ann.get("shape") == "polygon" and 
            ann.get("name") == "Filling" and 
            len(ann.get("annotations", [])) >= 4
            for ann in annotations
        )

        # Check if there are any valid filling annotations
        if not has_valid_filling:
            print(f"No valid filling annotations found for {entry['image_name']}")
            for ann in annotations:
                print(ann.get("shape"), ann.get("name"), len(ann.get("annotations", [])))
            continue

        if has_valid_filling:
            print(f"Visualizing: {entry['image_name']}")
            img_path = os.path.join(image_dir, entry["image_name"])

            # check if the image exists
            if os.path.exists(img_path):
                image = load_image(img_path)


                # Enhance the image using CLAHE
                enhanced_image = apply_clahe_color(image)
                # enhanced_image = image
                
                results = []
                y_true = []  # Ground truth
                y_pred = []  # Prediction
                
                # Process each annotation (each root filling polygon)
                for ann in annotations:
                    shape = ann.get("shape")
                    coords = ann.get("annotations", [])
                    name = ann.get("name")
                    if shape == "polygon" and name == "Filling" and len(coords) >= 4:
                        points = coords
                        # print(f"Processing annotation for {name}: {points}")

                        # Convert flat list to list of [x, y] pairs
                        points_xy = [[points[i], points[i+1]] for i in range(0, len(points), 2)]
                        scaled_points = scale_points(points_xy, image.shape[1], image.shape[0])
                        # print(f"Scaled points for {name}: {scaled_points}")

                        # Create the filled region mask
                        mask = create_mask_for_polygon(image.shape, scaled_points)

                        # Create border regions (inner and outer)
                        inner_border, outer_border = create_border_regions(mask, range_pixels)

                        # Analyze the average intensity in filling region and border regions
                        avg_filling, avg_inner, avg_outer = analyze_intensities(enhanced_image, mask, inner_border, outer_border)
                        results.append({
                            'label': name,
                            'avg_filling': avg_filling,
                            'avg_inner': avg_inner,
                            'avg_outer': avg_outer,
                            'mask': mask,
                            'inner_border': inner_border,
                            'outer_border': outer_border,
                            'scaled_points': scaled_points
                        })

                        
                        # Overlay the filled annotation region (with transparency) on the enhanced image
                        annotated_img = overlay_annotation(enhanced_image, scaled_points, color=(0, 255, 0), alpha=0.4)
                        
                        # Highlight the outer edge region in red using overlay_mask
                        outer_edge_overlay = overlay_mask(enhanced_image, outer_border, color=(255, 0, 0), alpha=0.4)
                        outer_edge_overlay = overlay_mask(enhanced_image, outer_border, color=(255, 0, 0), alpha=0.4)
                        
                        # Get count of low-intensity pixels
                        low_intensity_marked, low_intensity_pixel_count = mark_low_intensity_pixels(
                            enhanced_image, outer_border, threshold=160, mark_color=(0, 0, 255), alpha=0.6
                        )

                        # Count pixels in the filled annotation region
                        filled_pixel_count = np.sum(mask == 255)


                        # Display the images

                        if VISUALIZE_PLOTS:
                            plt.figure(figsize=(36, 10))
                            plt.subplot(1, 4, 1)
                            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            plt.title("Original Image")
                            plt.axis("off")
                            
                            plt.subplot(1, 4, 2)
                            plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                            plt.title(f"Annotation Filled ")
                            plt.axis("off")
                            
                            plt.subplot(1, 4, 3)
                            plt.imshow(cv2.cvtColor(outer_edge_overlay, cv2.COLOR_BGR2RGB))
                            plt.title(f"Outer Edge Region ")
                            plt.axis("off")
                            
                            # plt.subplot(1, 4, 4)
                            # plt.imshow(cv2.cvtColor(low_intensity_marked, cv2.COLOR_BGR2RGB))
                            # plt.title("Low Intensity Marked\n(Outer Edge)")
                            # plt.axis("off")
                            # plt.show()
                            
                            print(f"Results for:")
                            print(f"  Average intensity in filling region: {avg_filling:.2f}")
                            print(f"  Average intensity in inner border (edge inside): {avg_inner:.2f}")
                            print(f"  Average intensity in outer border (edge outside): {avg_outer:.2f}")
                            print("-" * 50)
                        
                        # Compute intensity profile
                        bins, intensity_profile, bin_masks = compute_intensity_profile(enhanced_image, mask, inner_pixels=8, outer_pixels=10)

                        # Step 1: Compute the mean intensity for bins 7 to 10
                        ref_bins = np.arange(7, 11)  # Bins from 7 to 10
                        # Use numpy for safe indexing
                        ref_indices = np.where(np.isin(bins, ref_bins))[0]
                        ref_intensities = [intensity_profile[i] for i in ref_indices]
                        mean_ref_intensity = np.mean(ref_intensities)

                        # Step 2: Identify bins where intensity is lower than mean_ref_intensity
                        composite_mask = np.zeros_like(mask)  # Initialize empty mask
                        for i, intensity in enumerate(intensity_profile):
                            bin_idx = bins[i]
                            if intensity < mean_ref_intensity - 4:  # Threshold for lower intensity
                                composite_mask = cv2.bitwise_or(composite_mask, bin_masks[bin_idx])

                        # Step 3: Identify pixels in composite border that have intensity < mean_ref_intensity
                        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
                        low_intensity_mask = (composite_mask == 255) & (gray < mean_ref_intensity)


                        # Step 4: Count and calculate percentage
                        low_intensity_pixel_count = np.sum(low_intensity_mask)
                        total_composite_pixels = np.sum(composite_mask == 255)
                        low_intensity_percentage = (low_intensity_pixel_count / total_composite_pixels) * 100 if total_composite_pixels > 0 else 0

                        if VISUALIZE_PLOTS:
                            print(f"Percentage of low-intensity pixels in composite border: {low_intensity_percentage:.2f}%")

                        # Compute the percentage
                        total_relevant_pixels = filled_pixel_count + low_intensity_pixel_count
                        low_intensity_percentage = (low_intensity_pixel_count / total_relevant_pixels) * 100 if total_relevant_pixels > 0 else 0

                        # print(filled_pixel_count, low_intensity_pixel_count, total_relevant_pixels, low_intensity_percentage)

                        if VISUALIZE_PLOTS:
                            print(f"Percentage of low-intensity pixels in outer region of : {low_intensity_percentage:.2f}%")

                        # Step 5: Visualize the composite mask and low-intensity pixels

                        if VISUALIZE_PLOTS:
                            marked = enhanced_image.copy()
                            marked[low_intensity_mask] = (0, 0, 255)  # Mark low-intensity pixels in red
                            plt.subplot(1, 4, 4)
                            plt.imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
                            plt.title("Low Intensity Pixels in Composite Border")
                            plt.axis("off")
                            plt.show()
                            
                            # Plot the intensity profile
                            plt.figure(figsize=(8, 5))
                            plt.plot(bins, intensity_profile, marker='o', linestyle='-')
                            plt.xlabel('Distance (pixels)\n(Negative: inside, 0: boundary, Positive: outside)')
                            plt.ylabel('Average Grayscale Intensity')
                            plt.title(f'Intensity Profile Across Border')
                            plt.grid(True)
                            plt.show()

                        # Step 6: Predict lateral seal
                        if low_intensity_percentage > 0:
                            pred_label = "incorrect"
                            y_pred.append("Incorrect")
                        else:
                            pred_label = "correct"
                            y_pred.append("Correct")
                        
                        # check ground truth
                        gt = ann.get("lateral_seal", "").strip().lower()
                        if gt == "correct":
                            y_true.append("Correct")
                        elif gt == "incorrect":
                            y_true.append("Incorrect")
                        else:
                            continue  # skip if no label
    # Compute metrics
    from collections import Counter
    def compute_metrics(y_true, y_pred, label):
        tp = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1, tp, fp, fn
    correct_prec, correct_rec, correct_f1, tp1, fp1, fn1 = compute_metrics(y_true, y_pred, "Correct")
    incorr_prec, incorr_rec, incorr_f1, tp2, fp2, fn2 = compute_metrics(y_true, y_pred, "Incorrect")
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true) if y_true else 0
    print("\nLateral Seal Classification Metrics:")
    print(f"Correct:    Precision={correct_prec:.3f}, Recall={correct_rec:.3f}, F1={correct_f1:.3f}, TP={tp1}, FP={fp1}, FN={fn1}")
    print(f"Incorrect:  Precision={incorr_prec:.3f}, Recall={incorr_rec:.3f}, F1={incorr_f1:.3f}, TP={tp2}, FP={fp2}, FN={fn2}")
    print(f"Overall Accuracy: {accuracy:.3f} ({sum(yt == yp for yt, yp in zip(y_true, y_pred))}/{len(y_true)})")
                        

if __name__ == "__main__":
    main()
