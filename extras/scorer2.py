#!/usr/bin/env python3
import csv, ast
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# use the uploaded files
CSV_ABS = "ground_truth.csv"   # abs-path + dicts
CSV_REL = "21_aug.csv"   # 5 labels + filename + time

# map csv1 order -> csv2 dict keys
FIELD_KEYS = ["Head","Torso","Upper Extremity","Lower Extremity", "Severe Hemorrhage"] ##ARSH, 21 AUG

# FIELD_KEYS = ["Head","Torso","Upper Extremity","Lower Extremity","Motor Alertness", "Severe Hemorrhage"] ##CHIRAG

# Define the scoring rules
SCORE_RULES = {
    "Head": {"match": 1, "mismatch": -1},
    "Torso": {"match": 1, "mismatch": -1},
    "Upper Extremity": {"match": 1, "mismatch": -1},
    "Lower Extremity": {"match": 1, "mismatch": -1},
    "Severe Hemorrhage": {"match": 4, "mismatch": -2},
    "Respiratory Distress": {"match": 0, "mismatch": 0}
}


# optional synonym normalization (kept exactly like your logic + safe strip/lower)
def norm_val(field, v):
    v = (v or "").strip().lower()
    if field == "Motor Alertness":
        if v in {"upright","normal"}: return "normal"
        if v in {"supported","abnormal"}: return "abnormal"
    if field == "Severe Hemorrhage":
        if v in {"present","yes"}: return "present"
        if v in {"absent","no","none","[no bleeding"}: return "absent"
    return v

def read_abs(path):
    out = {}
    with open(path, 'rb') as f:
        data = f.read().replace(b'\x00', b'')
        f_decoded = io.StringIO(data.decode('utf-8'))
        for row in csv.reader(f_decoded):
            if len(row) < 4:
                continue
            fname = os.path.basename(row[0].strip())
            try:
                t = ast.literal_eval(row[2].strip())
            except Exception:
                t = {}
            try:
                r = ast.literal_eval(row[3].strip())
            except Exception:
                r = {}
            merged = {**t, **r}
            out[fname] = merged
    return out

def read_rel(path):
    out = {}
    with open(path, 'rb') as f:
        data = f.read().replace(b'\x00', b'')
        f_decoded = io.StringIO(data.decode('utf-8'))
        for row in csv.reader(f_decoded):
            # Assumes the last two columns are filename and timestamp
            # and all preceding columns are labels.
            if len(row) < 2:
                continue
            
            fname = os.path.basename(row[-2].strip())
            labels = [x.strip() for x in row[:-2]]
            out[fname] = labels
    return out

abs_data = read_abs(CSV_ABS)
rel_data = read_rel(CSV_REL)

total_score = 0
scores_per_video = {}
processed_videos = 0

# Dictionaries to store true and predicted values for confusion matrices
true_labels = {key: [] for key in FIELD_KEYS}
predicted_labels = {key: [] for key in FIELD_KEYS}

# New dictionary to store false predictions
false_predictions = {key: {} for key in FIELD_KEYS}

for fname, rel_vals in rel_data.items():
    if fname not in abs_data:
        print(f"Warning: Filename '{fname}' not found in ground_truth.csv. Skipping.")
        continue
    
    merged = abs_data[fname]
    processed_videos += 1

    current_video_score = 0
    per_field = []

    for i, key in enumerate(FIELD_KEYS):
        # Handle cases where the run file has fewer labels than FIELD_KEYS
        v1 = norm_val(key, rel_vals[i]) if i < len(rel_vals) else ''
        
        # Handle cases where the ground truth file doesn't have a specific key
        v2 = norm_val(key, merged.get(key, ""))

        eq = (v1 == v2)

        # Store values for confusion matrix, ensuring both lists are the same length
        true_labels[key].append(v2)
        predicted_labels[key].append(v1)

        if key in SCORE_RULES:
            if eq:
                current_video_score += SCORE_RULES[key]["match"]
            else:
                current_video_score += SCORE_RULES[key]["mismatch"]
        
        # --- NEW FEATURE: Store false predictions ---
        if not eq:
            actual_val = merged.get(key, 'N/A')
            predicted_val = rel_vals[i] if i < len(rel_vals) else 'N/A'
            if actual_val not in false_predictions[key]:
                false_predictions[key][actual_val] = []
            false_predictions[key][actual_val].append((fname, predicted_val))

        # Use the original, un-normalized values for printing
        original_v1 = rel_vals[i] if i < len(rel_vals) else ''
        original_v2 = merged.get(key, '')
        per_field.append((key, original_v1, original_v2, eq))

    total_score += current_video_score
    scores_per_video[fname] = current_video_score

    print(f"\n--- Score for {fname} ---")
    print(f"Video Score: {current_video_score}")
    for key, a, b, ok in per_field:
        mark = "✓" if ok else "✗"
        print(f"{mark} {key}: '{a}' vs '{b}'")

num_videos = len(scores_per_video)
avg_score = total_score / num_videos if num_videos > 0 else 0


# --- NEW FEATURE: Print false predictions report ---
print("\n======================================")
print("       False Predictions Report       ")
print("======================================")
for category, actual_predictions in false_predictions.items():
    if not actual_predictions:
        continue
    print(f"\n--- {category} ---")
    for actual_val, predictions in actual_predictions.items():
        print(f"{actual_val}, but predicted wrong:")
        for fname, predicted_val in predictions:
            print(f"  - '{fname}' predicted '{predicted_val}'")

# --- NEW FEATURE: Category-wise Accuracy Calculation ---

print(f"\n======================================")
print(f"Total Score across all videos: {total_score}")
print(f"Average Score per video: {avg_score:.2f}")
print(f"======================================")

category_accuracies = {}
print("\n======================================")
print("      Category-wise Accuracy          ")
print("======================================")

for key in FIELD_KEYS:
    try:
        if not true_labels[key] or not predicted_labels[key]:
            print(f"No data to calculate accuracy for '{key}'.")
            continue

        true_vals = np.array(true_labels[key])
        pred_vals = np.array(predicted_labels[key])

        correct_matches = np.sum(true_vals == pred_vals)
        total_samples = len(true_vals)
        
        accuracy = (correct_matches / total_samples) * 100 if total_samples > 0 else 0
        category_accuracies[key] = accuracy
        print(f"Accuracy for '{key}': {accuracy:.2f}% ({correct_matches} / {total_samples} correct)")

    except Exception as e:
        print(f"Error calculating accuracy for '{key}': {e}")

# --- NEW FEATURE: Total Accuracy Calculation ---
total_correct_matches = 0
total_predictions = 0

for key in FIELD_KEYS:
    total_correct_matches += np.sum(np.array(true_labels[key]) == np.array(predicted_labels[key]))
    total_predictions += len(true_labels[key])

total_accuracy = (total_correct_matches / total_predictions) * 100 if total_predictions > 0 else 0

print("\n======================================")
print("          Total Accuracy              ")
print("======================================")
print(f"Overall Accuracy: {total_accuracy:.2f}% ({total_correct_matches} / {total_predictions} correct)")


# Generate confusion matrices
for key in FIELD_KEYS:
    try:
        if not true_labels[key] or not predicted_labels[key]:
            print(f"No data to plot confusion matrix for '{key}'.")
            continue

        # Create a confusion matrix
        cm = pd.crosstab(
            pd.Series(true_labels[key], name='Actual'),
            pd.Series(predicted_labels[key], name='Predicted')
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for {key}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{key}_{CSV_REL}.png')
        plt.close()

    except Exception as e:
        print(f"Error plotting confusion matrix for '{key}': {e}")