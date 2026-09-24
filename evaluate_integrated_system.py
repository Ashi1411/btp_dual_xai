import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support
)
from src.integrated_model import ExplainableSolarFaultDetector


def generate_scada_for_class(true_class: str) -> dict:
    """
    Generates realistic SCADA telemetry distributions with random environmental noise
    aligned with the true physical fault condition.
    """
    if true_class == "HEALTHY":
        irr = np.random.uniform(0.75, 1.0)
        mod_temp = np.random.uniform(30.0, 42.0)
        dc = irr * 200.0 * (1.0 - (mod_temp - 25.0) * 0.004) * np.random.uniform(0.96, 1.02)
        ac = dc * np.random.uniform(0.93, 0.96)
    
    elif true_class == "CRACKED":
        irr = np.random.uniform(0.70, 1.0)
        mod_temp = np.random.uniform(35.0, 48.0)
        base_dc = irr * 200.0 * (1.0 - (mod_temp - 25.0) * 0.004)
        dc = base_dc * np.random.uniform(0.50, 0.75)  # 25-50% drop due to cell disconnection
        ac = dc * np.random.uniform(0.90, 0.95)

    elif true_class == "HOTSPOT":
        irr = np.random.uniform(0.70, 1.0)
        mod_temp = np.random.uniform(58.0, 85.0)       # Elevated overheating
        base_dc = irr * 200.0 * (1.0 - (mod_temp - 25.0) * 0.004)
        dc = base_dc * np.random.uniform(0.65, 0.85)  # Moderate thermal loss drop
        ac = dc * np.random.uniform(0.90, 0.94)

    elif true_class == "SHADOW":
        irr = np.random.uniform(0.10, 0.38)           # Low irradiance
        mod_temp = np.random.uniform(22.0, 32.0)
        base_dc = irr * 200.0 * (1.0 - (mod_temp - 25.0) * 0.004)
        dc = base_dc * np.random.uniform(0.85, 0.98)
        ac = dc * np.random.uniform(0.88, 0.93)
    
    else:
        # Default nominal fallback
        irr, mod_temp, dc, ac = 0.8, 35.0, 150.0, 140.0

    return {
        "DC_POWER": round(float(dc), 2),
        "AC_POWER": round(float(ac), 2),
        "AMBIENT_TEMPERATURE": round(float(np.random.uniform(22.0, 32.0)), 2),
        "MODULE_TEMPERATURE": round(float(mod_temp), 2),
        "IRRADIATION": round(float(irr), 2)
    }


def run_large_scale_evaluation(test_dir: str):
    print("==================================================")
    print("    LARGE-SCALE MULTIMODAL INTEGRATED EVALUATION  ")
    print("==================================================\n")

    detector = ExplainableSolarFaultDetector(
        scada_model_path="models/xgboost_scada.joblib",
        scaler_path="models/scada_scaler.joblib",
        vision_model_path="models/resnet18_thermal_best.pth",
        label_encoder_path="models/label_encoder.joblib"
    )

    classes = ["HEALTHY", "CRACKED", "HOTSPOT", "SHADOW"]
    y_true = []
    y_pred_integrated = []
    y_pred_vision_only = []

    records = []

    for c in classes:
        # Search for images in extensions (.jpg, .png, .jpeg)
        folder = os.path.join(test_dir, c)
        if not os.path.exists(folder):
            print(f"[Warning] Folder missing: {folder}")
            continue

        images = glob.glob(os.path.join(folder, "*.jpg")) + \
                 glob.glob(os.path.join(folder, "*.png")) + \
                 glob.glob(os.path.join(folder, "*.jpeg"))
        
        print(f"Found {len(images)} test samples for class '{c}'. Processing...")

        for img_path in images:
            telemetry = generate_scada_for_class(c)
            
            # Integrated inference
            res = detector.evaluate_panel(scada_input=telemetry, thermal_image_input=img_path)
            
            pred_integrated = res["prediction"]
            pred_vision = res["explainable_ai"]["vision_metrics"].get("primary_detected_class", "UNKNOWN")

            y_true.append(c)
            y_pred_integrated.append(pred_integrated)
            y_pred_vision_only.append(pred_vision)

            records.append({
                "image_path": img_path,
                "true_label": c,
                "integrated_prediction": pred_integrated,
                "vision_prediction": pred_vision,
                **telemetry
            })

    if not y_true:
        print("Error: No test images found. Please verify your test directory path.")
        return

    # Save complete dataset records
    results_df = pd.DataFrame(records)
    results_df.to_csv("multimodal_large_test_results.csv", index=False)
    print(f"\n[SUCCESS] Processed Total Samples: N = {len(y_true)}")
    print("Detailed inference records exported to 'multimodal_large_test_results.csv'.\n")

    # Metrics calculation
    acc_int = accuracy_score(y_true, y_pred_integrated)
    acc_vis = accuracy_score(y_true, y_pred_vision_only)
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred_integrated, average="macro", zero_division=0
    )

    print("--------------------------------------------------")
    print("    INTEGRATED SYSTEM METRICS (N = {})".format(len(y_true)))
    print("--------------------------------------------------")
    print(f"Integrated System Accuracy : {acc_int * 100:.2f}%")
    print(f"Vision-Only Accuracy      : {acc_vis * 100:.2f}%")
    print(f"Multimodal Fusion Lift    : +{(acc_int - acc_vis) * 100:.2f}%")
    print(f"Macro Precision           : {p_macro * 100:.2f}%")
    print(f"Macro Recall              : {r_macro * 100:.2f}%")
    print(f"Macro F1-Score            : {f1_macro * 100:.2f}%")
    print("--------------------------------------------------\n")

    print("--- Detailed Integrated Classification Report ---")
    print(classification_report(y_true, y_pred_integrated, target_names=classes, zero_division=0))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred_integrated, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Integrated Multimodal System Confusion Matrix (N={len(y_true)})")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Ground Truth Class")
    plt.tight_layout()
    plt.savefig("large_scale_confusion_matrix.png", dpi=300)
    print("Confusion Matrix plot saved as 'large_scale_confusion_matrix.png'.")

if __name__ == "__main__":
    # Point this to your actual image test set directory
    # Structure expected: dataset_dir/HEALTHY, dataset_dir/CRACKED, etc.
    TEST_DATASET_DIR = "data/test"  # <-- Update path to your test folder
    run_large_scale_evaluation(TEST_DATASET_DIR)