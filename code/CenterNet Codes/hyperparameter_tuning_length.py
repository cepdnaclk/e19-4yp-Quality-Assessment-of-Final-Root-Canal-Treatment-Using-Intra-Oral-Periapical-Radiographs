import numpy as np
import json
import os
from check_root_filling_length import (
    load_annotations,
    parse_coordinates,
    analyze_filling,
    get_ground_truth_label,
    calculate_metrics
)

# Hyperparameter ranges
ADEQUATE_RANGE = np.arange(8.0, 16.01, 0.2)
OVERFILL_RANGE = np.arange(0.1, 1.01, 0.1)

json_file = "annotated_and_labeled.json"
image_dir = "annotated_and_labeled"

# Load annotations once
data = load_annotations(json_file)

results = []

for adequate_thresh in ADEQUATE_RANGE:
    for overfill_thresh in OVERFILL_RANGE:
        # Patch the global thresholds in the imported module
        import check_root_filling_length as crfl
        crfl.ADEQUATE_THRESHOLD_PERCENT = adequate_thresh
        crfl.OVERFILL_THRESHOLD_PERCENT = overfill_thresh

        all_predictions = []
        all_ground_truths = []
        for entry in data:
            image_name = entry.get("image_name", "Unknown")
            annotations = entry.get("annotation", [])
            fillings = []
            apices = []
            for ann in annotations:
                if ann.get("name") == "Filling" and ann.get("shape") == "polygon":
                    coords = parse_coordinates(ann.get("annotations", []))
                    if len(coords) >= 3:
                        fillings.append({
                            'id': ann.get("id", 0),
                            'coords': coords,
                            'annotation': ann
                        })
                elif ann.get("name") == "Root Apex" and ann.get("shape") == "dot":
                    coords = parse_coordinates(ann.get("annotations", []))
                    if len(coords) == 1:
                        apices.append(coords[0])
            if not fillings or not apices:
                continue
            for filling in fillings:
                try:
                    result = analyze_filling(filling['coords'], apices, filling['id'], image_name)
                    ground_truth = get_ground_truth_label(filling['annotation'])
                    if ground_truth != "Unknown":
                        all_predictions.append(result['classification'])
                        all_ground_truths.append(ground_truth)
                except Exception:
                    continue
        if all_predictions and all_ground_truths:
            metrics = calculate_metrics(all_predictions, all_ground_truths)
            overall_acc = metrics['overall']['accuracy']
            avg_f1 = np.mean([metrics[c]['f1_score'] for c in ['Adequate', 'Underfilled', 'Overfilled']])
            results.append({
                'adequate_thresh': adequate_thresh,
                'overfill_thresh': overfill_thresh,
                'accuracy': overall_acc,
                'avg_f1': avg_f1,
                'metrics': metrics
            })
        else:
            results.append({
                'adequate_thresh': adequate_thresh,
                'overfill_thresh': overfill_thresh,
                'accuracy': 0,
                'avg_f1': 0,
                'metrics': None
            })

# Find the best by accuracy and F1
best_acc = max(results, key=lambda x: x['accuracy'])
best_f1 = max(results, key=lambda x: x['avg_f1'])

print("Best by Accuracy:")
print(f"  Adequate Threshold: {best_acc['adequate_thresh']}")
print(f"  Overfill Threshold: {best_acc['overfill_thresh']}")
print(f"  Accuracy: {best_acc['accuracy']:.4f}")
print(f"  Avg F1: {best_acc['avg_f1']:.4f}")
print(f"  Metrics: {best_acc['metrics']}")

print("\nBest by Avg F1:")
print(f"  Adequate Threshold: {best_f1['adequate_thresh']}")
print(f"  Overfill Threshold: {best_f1['overfill_thresh']}")
print(f"  Accuracy: {best_f1['accuracy']:.4f}")
print(f"  Avg F1: {best_f1['avg_f1']:.4f}")
print(f"  Metrics: {best_f1['metrics']}")

# Optionally, save all results to a file
with open("hyperparameter_tuning_length_results.json", "w") as f:
    json.dump(results, f, indent=2) 