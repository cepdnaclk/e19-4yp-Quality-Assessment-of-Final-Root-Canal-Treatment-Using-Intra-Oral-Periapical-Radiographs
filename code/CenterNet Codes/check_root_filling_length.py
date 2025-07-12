import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
import math
from collections import defaultdict

# Configuration
ADEQUATE_THRESHOLD_PERCENT = 9.3  # Configurable threshold for adequate fillings
OVERFILL_THRESHOLD_PERCENT = 0.5  # Configurable threshold for overfilled fillings
VISUALIZE_PLOTS = False # Set to True to visualize the plots
VISUALIZE_ONLY_WRONG_PREDS = False # Set to True to visualize only wrong predictions

def load_annotations(json_file: str) -> List[Dict[str, Any]]:
    """Load the annotation JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def parse_coordinates(annotations: List[int]) -> List[Tuple[int, int]]:
    """Parse flat list of coordinates into [x, y] pairs."""
    coordinates = []
    for i in range(0, len(annotations), 2):
        if i + 1 < len(annotations):
            coordinates.append((annotations[i], annotations[i + 1]))
    return coordinates

def calculate_euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_filling_extremes(filling_coords: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Find the points with minimum and maximum Y-coordinates (top and bottom of filling)."""
    if not filling_coords:
        raise ValueError("No coordinates provided for filling")
    
    y_min_point = min(filling_coords, key=lambda p: p[1])
    y_max_point = max(filling_coords, key=lambda p: p[1])
    
    return y_min_point, y_max_point

def match_filling_to_apex(filling_coords: List[Tuple[int, int]], apex_points: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """
    Match filling to the closest apex and determine apical and coronal ends.
    Returns: (apical_end, coronal_end, matched_apex_index)
    """
    if not apex_points:
        raise ValueError("No apex points available for matching")
    
    y_min_point, y_max_point = find_filling_extremes(filling_coords)
    
    # Calculate distances from both filling ends to all apices
    min_dist = float('inf')
    matched_apex_idx = 0
    apical_end = y_min_point
    coronal_end = y_max_point
    
    for i, apex in enumerate(apex_points):
        dist_to_min = calculate_euclidean_distance(y_min_point, apex)
        dist_to_max = calculate_euclidean_distance(y_max_point, apex)
        
        # Find the closest distance and corresponding filling end
        if dist_to_min < dist_to_max:
            if dist_to_min < min_dist:
                min_dist = dist_to_min
                matched_apex_idx = i
                apical_end = y_min_point
                coronal_end = y_max_point
        else:
            if dist_to_max < min_dist:
                min_dist = dist_to_max
                matched_apex_idx = i
                apical_end = y_max_point
                coronal_end = y_min_point
    
    return apical_end, coronal_end, matched_apex_idx

def determine_overfill_condition(apical_end: Tuple[int, int], coronal_end: Tuple[int, int], matched_apex: Tuple[int, int]) -> bool:
    """
    Determine if the filling is overfilled based on orientation.
    Returns True if overfilled, False otherwise.
    """
    # Determine tooth orientation based on Y-coordinates
    if apical_end[1] > coronal_end[1]:  # Lower jaw tooth (apical end is lower)
        return apical_end[1] > matched_apex[1]  # Overfill if apical end goes past apex
    else:  # Upper jaw tooth (apical end is higher)
        return apical_end[1] < matched_apex[1]  # Overfill if apical end goes past apex

def classify_filling_length(gap_distance: float, filling_length: float, is_overfilled: bool) -> Tuple[str, float]:
    """
    Classify the filling length based on gap distance and overfill condition.
    Returns: (classification, relative_distance_percent)
    """
    relative_distance_percent = (gap_distance / filling_length) * 100 if filling_length > 0 else 0
    
    if is_overfilled and relative_distance_percent >= OVERFILL_THRESHOLD_PERCENT:
        return "Overfilled", relative_distance_percent
    elif relative_distance_percent <= ADEQUATE_THRESHOLD_PERCENT:
        return "Adequate", relative_distance_percent
    else:
        return "Underfilled", relative_distance_percent

def calculate_centerline_length(filling_coords: List[Tuple[int, int]]) -> float:
    """
    Calculates the length of a filling by tracing its centerline, accounting for curvature.

    This method works by:
    1. Iterating vertically (Y-axis) through the polygon from its top to its bottom.
    2. At each 1-pixel vertical step (scanline), it finds all intersection points
       with the polygon's edges.
    3. It calculates the midpoint of the outermost intersections for that scanline.
    4. These midpoints form the "centerline" of the filling.
    5. The function then sums the distances between consecutive points on this
       centerline to get the total curved length.
    """
    if len(filling_coords) < 3:
        return 0.0

    # Sort coordinates to ensure proper edge traversal
    sorted_coords = sorted(filling_coords, key=lambda p: p[1])
    y_min, y_max = sorted_coords[0][1], sorted_coords[-1][1]

    centerline_points = []
    # Iterate through each vertical pixel of the polygon
    for y in range(y_min, y_max + 1):
        intersections = []
        # Find all intersections of the horizontal scanline with the polygon edges
        for i in range(len(filling_coords)):
            p1 = filling_coords[i]
            p2 = filling_coords[(i + 1) % len(filling_coords)]

            # Check if the scanline y is between the y-coordinates of the edge points
            if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                # Calculate the x-coordinate of the intersection
                x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                intersections.append(x)

        if intersections:
            # Get the midpoint of the leftmost and rightmost intersections
            x_min = min(intersections)
            x_max = max(intersections)
            centerline_points.append(((x_min + x_max) / 2, y))

    if not centerline_points:
        return 0.0, []

    # Calculate the total length of the centerline by summing segment distances
    total_length = 0.0
    for i in range(len(centerline_points) - 1):
        total_length += calculate_euclidean_distance(centerline_points[i], centerline_points[i+1])

    return total_length, centerline_points

def analyze_filling(filling_coords: List[Tuple[int, int]], apex_points: List[Tuple[int, int]], 
                      filling_id: int, image_name: str) -> Dict[str, Any]:
    """
    Analyze a single filling and return classification results.
    """
    # Match filling to apex
    apical_end, coronal_end, matched_apex_idx = match_filling_to_apex(filling_coords, apex_points)
    matched_apex = apex_points[matched_apex_idx]
    
    # Calculate measurements
    # Use the new, more accurate centerline length calculation
    filling_length, centerline = calculate_centerline_length(filling_coords)
    gap_distance = calculate_euclidean_distance(apical_end, matched_apex)
    
    # Determine overfill condition
    is_overfilled = determine_overfill_condition(apical_end, coronal_end, matched_apex)
    
    # Classify filling
    classification, relative_distance_percent = classify_filling_length(gap_distance, filling_length, is_overfilled)
    
    return {
        'filling_id': filling_id,
        'matched_apex_id': matched_apex_idx,
        'classification': classification,
        'relative_distance_percent': relative_distance_percent,
        'gap_distance': gap_distance,
        'filling_length': filling_length,
        'apical_end': apical_end,
        'coronal_end': coronal_end,
        'matched_apex': matched_apex,
        'is_overfilled': is_overfilled,
        'centerline': centerline  # Add the centerline points to the result
    }

def get_ground_truth_label(filling_annotation: Dict[str, Any]) -> str:
    """Extract ground truth label from the annotation."""
    ground_truth = filling_annotation.get("filling_length", "").strip()
    
    # Map ground truth labels to our classification categories
    if ground_truth.lower() in ["correct", "adequate"]:
        return "Adequate"
    elif ground_truth.lower() in ["underfilled", "underfill"]:
        return "Underfilled"
    elif ground_truth.lower() in ["overfilled", "overfill"]:
        return "Overfilled"
    else:
        return "Unknown"  # For cases where ground truth is not available

def calculate_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Calculate precision, recall, and accuracy for each class.
    """
    classes = ["Adequate", "Underfilled", "Overfilled"]
    metrics = {}
    
    for class_name in classes:
        # True Positives: predicted as class and actually class
        tp = sum(1 for pred, gt in zip(predictions, ground_truths) 
                if pred == class_name and gt == class_name)
        
        # False Positives: predicted as class but actually not class
        fp = sum(1 for pred, gt in zip(predictions, ground_truths) 
                if pred == class_name and gt != class_name)
        
        # False Negatives: not predicted as class but actually class
        fn = sum(1 for pred, gt in zip(predictions, ground_truths) 
                if pred != class_name and gt == class_name)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Overall accuracy
    correct_predictions = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    total_predictions = len(predictions)
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    metrics['overall'] = {
        'accuracy': overall_accuracy,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions
    }
    
    return metrics

def visualize_analysis(image_path: str, filling_coords: List[Tuple[int, int]], 
                      analysis_result: Dict[str, Any], image_name: str):
    """
    Create a visualization of the analysis results.
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Define colors based on classification
        color_map = {
            "Adequate": "green",
            "Underfilled": "yellow", 
            "Overfilled": "red"
        }
        fill_color = color_map.get(analysis_result['classification'], "blue")
        
        # Draw filling polygon
        filling_poly = patches.Polygon(filling_coords, closed=True, 
                                     fill=True, alpha=0.3, 
                                     facecolor=fill_color, edgecolor='black', linewidth=2)
        ax.add_patch(filling_poly)
        
        # Mark apical end
        apical_end = analysis_result['apical_end']
        ax.plot(apical_end[0], apical_end[1], 'ro', markersize=10, label='Apical End')
        
        # Mark matched apex
        matched_apex = analysis_result['matched_apex']
        ax.plot(matched_apex[0], matched_apex[1], 'bo', markersize=10, label='Root Apex')
        
        # Draw line between apical end and apex
        ax.plot([apical_end[0], matched_apex[0]], [apical_end[1], matched_apex[1]], 
                'k--', linewidth=2, alpha=0.7)

        # Plot the calculated centerline
        centerline = analysis_result.get('centerline', [])
        if centerline:
            # Unzip the list of tuples into two lists: x_coords and y_coords
            centerline_x, centerline_y = zip(*centerline)
            ax.plot(centerline_x, centerline_y, 'c', linewidth=2, label='Calculated Centerline') # 'c--' is a cyan line
        
        # Add text with classification
        classification = analysis_result['classification']
        percentage = analysis_result['relative_distance_percent']
        
        if classification == "Overfilled":
            text = f"{classification}\nOver-extension: {percentage:.1f}%"
        else:
            text = f"{classification}\nGap: {percentage:.1f}%"
        
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f"Root Canal Filling Analysis - {image_name}")
        ax.legend()
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Warning: Could not create visualization for {image_name}: {str(e)}")

def main():
    """Main function to analyze root canal filling length."""
    json_file = "annotated_and_labeled.json"
    image_dir = "annotated_and_labeled"
    
    # Load annotations
    data = load_annotations(json_file)
    print(f"Loaded {len(data)} entries from {json_file}")
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    total_fillings = 0
    classification_counts = {"Adequate": 0, "Underfilled": 0, "Overfilled": 0}
    
    # Lists to store predictions and ground truths for comparison
    all_predictions = []
    all_ground_truths = []
    comparison_data = []
    
    for entry in data:
        image_name = entry.get("image_name", "Unknown")
        annotations = entry.get("annotation", [])
        
        # Separate fillings and apices
        fillings = []
        apices = []
        
        for ann in annotations:
            if ann.get("name") == "Filling" and ann.get("shape") == "polygon":
                coords = parse_coordinates(ann.get("annotations", []))
                if len(coords) >= 3:  # Need at least 3 points for a polygon
                    fillings.append({
                        'id': ann.get("id", 0),
                        'coords': coords,
                        'annotation': ann  # Keep the original annotation for ground truth
                    })
            
            elif ann.get("name") == "Root Apex" and ann.get("shape") == "dot":
                coords = parse_coordinates(ann.get("annotations", []))
                if len(coords) == 1:  # Single point for apex
                    apices.append(coords[0])
        
        # Skip if no fillings or apices
        if not fillings or not apices:
            print(f"Image: {image_name} | Skipped: No fillings or apices found")
            continue
        
        # Analyze each filling
        for filling in fillings:
            try:
                result = analyze_filling(filling['coords'], apices, filling['id'], image_name)
                
                # Get ground truth label
                ground_truth = get_ground_truth_label(filling['annotation'])
                
                # Store for comparison
                if ground_truth != "Unknown":
                    all_predictions.append(result['classification'])
                    all_ground_truths.append(ground_truth)
                    comparison_data.append({
                        'image_name': image_name,
                        'filling_id': filling['id'],
                        'predicted': result['classification'],
                        'ground_truth': ground_truth,
                        'percentage': result['relative_distance_percent']
                    })
                
                # Print results with ground truth comparison
                classification = result['classification']
                percentage = result['relative_distance_percent']
                matched_apex_id = result['matched_apex_id']
                
                if classification == "Overfilled":
                    print(f"Image: {image_name} | Filling ID: {filling['id']} | Matched Apex ID: {matched_apex_id} | Status: {classification} (Over-extension: {percentage:.1f}% of filling length) | Ground Truth: {ground_truth}")
                else:
                    print(f"Image: {image_name} | Filling ID: {filling['id']} | Matched Apex ID: {matched_apex_id} | Status: {classification} (Gap: {percentage:.1f}% of filling length) | Ground Truth: {ground_truth}")
                
                # Update counts
                classification_counts[classification] += 1
                total_fillings += 1
                
                # Conditional visualization
                image_path = os.path.join(image_dir, image_name)
                if os.path.exists(image_path):
                    if VISUALIZE_PLOTS and not VISUALIZE_ONLY_WRONG_PREDS:
                        # Visualize all if VISUALIZE_PLOTS is True and VISUALIZE_ONLY_WRONG_PREDS is False
                        visualize_analysis(image_path, filling['coords'], result, image_name)
                    elif VISUALIZE_ONLY_WRONG_PREDS and result['classification'] != ground_truth:
                        # Visualize only wrong predictions if VISUALIZE_ONLY_WRONG_PREDS is True
                        visualize_analysis(image_path, filling['coords'], result, image_name)
                
            except Exception as e:
                print(f"Image: {image_name} | Filling ID: {filling['id']} | Error: {str(e)}")
    
    # Calculate metrics
    if all_predictions and all_ground_truths:
        metrics = calculate_metrics(all_predictions, all_ground_truths)
        
        # Print detailed comparison
        print("\n" + "="*80)
        print("DETAILED COMPARISON (Predicted vs Ground Truth)")
        print("="*80)
        
        correct_predictions = 0
        for item in comparison_data:
            status = "✓" if item['predicted'] == item['ground_truth'] else "✗"
            correct_predictions += 1 if item['predicted'] == item['ground_truth'] else 0
            print(f"{status} {item['image_name']} | Filling {item['filling_id']} | Predicted: {item['predicted']} | Ground Truth: {item['ground_truth']} | Gap/Over-extension: {item['percentage']:.1f}%")
        
        print(f"\nCorrect Predictions: {correct_predictions}/{len(comparison_data)}")
        
        # Print metrics
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        
        for class_name in ["Adequate", "Underfilled", "Overfilled"]:
            if class_name in metrics:
                class_metrics = metrics[class_name]
                print(f"\n{class_name}:")
                print(f"   Precision: {class_metrics['precision']:.3f} ({class_metrics['tp']}/({class_metrics['tp']}+{class_metrics['fp']}))")
                print(f"   Recall: {class_metrics['recall']:.3f} ({class_metrics['tp']}/({class_metrics['tp']}+{class_metrics['fn']}))")
                print(f"   F1-Score: {class_metrics['f1_score']:.3f}")
        
        overall = metrics['overall']
        print(f"\nOverall Accuracy: {overall['accuracy']:.3f} ({overall['correct_predictions']}/{overall['total_predictions']})")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total fillings analyzed: {total_fillings}")
    print(f"Adequate fillings: {classification_counts['Adequate']}")
    print(f"Underfilled fillings: {classification_counts['Underfilled']}")
    print(f"Overfilled fillings: {classification_counts['Overfilled']}")
    
    if total_fillings > 0:
        print(f"\nPercentage breakdown:")
        print(f"Adequate: {(classification_counts['Adequate']/total_fillings)*100:.1f}%")
        print(f"Underfilled: {(classification_counts['Underfilled']/total_fillings)*100:.1f}%")
        print(f"Overfilled: {(classification_counts['Overfilled']/total_fillings)*100:.1f}%")

if __name__ == "__main__":
    main()