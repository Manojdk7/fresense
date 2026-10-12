"""
Model Evaluation and Benchmark Pipeline (Phase 2.3)
Evaluates pre-trained MobileNetV2 on real test dataset images from 01_datasets/.
Calculates precision, recall, F1-score, accuracy, and confusion matrix.
Saves metrics to 06_results/metrics.json.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
TEST_DATASET_DIR = BASE_DIR / "01_datasets" / "images" / "fresh_rotten_classification" / "dataset" / "Test"
RESULTS_DIR = BASE_DIR / "06_results"
MODEL_PATH = BASE_DIR / "02_models" / "food_spoilage_model_best.pth"

if not MODEL_PATH.exists():
    fallback = Path(r"C:\Users\lenovo\Downloads\food_spoilage_model_best.pth")
    if fallback.exists():
        MODEL_PATH = fallback

# Normalization mapping for folder typos in dataset to model classes
FOLDER_TO_MODEL_CLASS = {
    "freshapples": "freshapples",
    "freshbanana": "freshbanana",
    "freshcucumber": "freshcucumber",
    "freshokra": "freshokra",
    "freshoranges": "freshoranges",
    "freshpatato": "freshpotato",
    "freshtamto": "freshtomato",
    "rottenapples": "rottenapples",
    "rottenbanana": "rottenbanana",
    "rottencucumber": "rottencucumber",
    "rottenokra": "rottenokra",
    "rottenoranges": "rottenoranges",
    "rottenpatato": "rottenpotato",
    "rottentamto": "rottentomato",
}


def load_model(model_path):
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    best_val_acc = checkpoint.get("best_val_acc", None)

    model = models.mobilenet_v2(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1280, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, class_names, transform, best_val_acc


def evaluate(sample_per_class=35, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if not TEST_DATASET_DIR.exists():
        raise FileNotFoundError(f"Test dataset directory not found at {TEST_DATASET_DIR}")

    model, class_names, transform, best_val_acc = load_model(MODEL_PATH)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    print(f"Found {len(class_names)} model classes.")
    test_folders = [f for f in os.listdir(TEST_DATASET_DIR) if os.path.isdir(TEST_DATASET_DIR / f)]
    print(f"Found {len(test_folders)} test folders. Evaluating up to {sample_per_class} per class...")

    # Metrics trackers
    y_true_binary = []
    y_pred_binary = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    total_evaluated = 0

    with torch.no_grad():
        for folder in sorted(test_folders):
            mapped_class = FOLDER_TO_MODEL_CLASS.get(folder, folder)
            if mapped_class not in class_to_idx:
                continue

            target_idx = class_to_idx[mapped_class]
            is_true_rotten = "rotten" in mapped_class.lower()

            folder_path = TEST_DATASET_DIR / folder
            all_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            
            sample_files = random.sample(all_files, min(sample_per_class, len(all_files))) if sample_per_class else all_files

            for fname in sample_files:
                img_path = folder_path / fname
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = transform(img).unsqueeze(0)
                    output = model(tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    pred_class = class_names[pred_idx]

                    # Rotten probability for binary freshness
                    rotten_prob = sum(probs[i].item() for i, name in enumerate(class_names) if "rotten" in name.lower())
                    is_pred_rotten = rotten_prob >= 0.5

                    # Record
                    is_match = (pred_idx == target_idx)
                    class_total[mapped_class] += 1
                    if is_match:
                        class_correct[mapped_class] += 1

                    y_true_binary.append(1 if is_true_rotten else 0)
                    y_pred_binary.append(1 if is_pred_rotten else 0)

                    confusion[mapped_class][pred_class] += 1
                    total_evaluated += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    # Compute binary metrics (Fresh vs Rotten)
    # 0 = Fresh, 1 = Rotten
    tp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 0)

    binary_accuracy = round((tp + tn) / max(1, (tp + tn + fp + fn)) * 100, 2)
    binary_precision = round(tp / max(1, (tp + fp)), 4) if (tp + fp) else 0.0
    binary_recall = round(tp / max(1, (tp + fn)), 4) if (tp + fn) else 0.0
    binary_f1 = round(2 * (binary_precision * binary_recall) / max(1e-6, (binary_precision + binary_recall)), 4) if (binary_precision + binary_recall) else 0.0

    # Per-class metrics
    per_class_metrics = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for cls in sorted(class_total.keys()):
        tot = class_total[cls]
        correct = class_correct[cls]
        rec = correct / tot if tot else 0.0

        # Class precision: true positives / predicted positives
        predicted_as_cls = sum(confusion[c][cls] for c in class_total.keys())
        prec = correct / predicted_as_cls if predicted_as_cls else 0.0

        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        macro_precisions.append(prec)
        macro_recalls.append(rec)
        macro_f1s.append(f1)

        per_class_metrics[cls] = {
            "test_samples": tot,
            "accuracy": round(correct / tot * 100, 1) if tot else 0.0,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

    overall_accuracy = round(sum(class_correct.values()) / max(1, total_evaluated) * 100, 2)
    macro_f1 = round(sum(macro_f1s) / max(1, len(macro_f1s)), 4)
    macro_prec = round(sum(macro_precisions) / max(1, len(macro_precisions)), 4)
    macro_rec = round(sum(macro_recalls) / max(1, len(macro_recalls)), 4)

    # Convert confusion matrix to JSON serializable dict
    confusion_dict = {cls: dict(confusion[cls]) for cls in class_total.keys()}

    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": "MobileNetV2 Transfer Learning (Fine-Tuned)",
        "backbone": "MobileNetV2",
        "num_classes": len(class_names),
        "total_test_samples_evaluated": total_evaluated,
        "overall_multiclass_accuracy": overall_accuracy,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1_score": macro_f1,
        "binary_freshness": {
            "accuracy": binary_accuracy,
            "precision": binary_precision,
            "recall": binary_recall,
            "f1_score": binary_f1,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "training_checkpoint_val_acc": best_val_acc,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_dict,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSuccessfully evaluated {total_evaluated} test images.")
    print(f"Overall Multiclass Accuracy: {overall_accuracy}%")
    print(f"Binary Freshness Accuracy:   {binary_accuracy}% (F1: {binary_f1})")
    print(f"Macro F1-Score:              {macro_f1}")
    print(f"Saved benchmark metrics to:  {out_path}")

    return metrics_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 food freshness model")
    parser.add_argument("--samples", type=int, default=35, help="Number of samples to evaluate per test class (default: 35)")
    args = parser.parse_args()
    evaluate(sample_per_class=args.samples)
