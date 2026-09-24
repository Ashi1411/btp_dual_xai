import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import time

try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExplainableSolarFaultDetector:
    """
    Dual-Stream Explainable AI Engine combining SCADA telemetry (XGBoost)
    and Thermal IR Vision (ResNet-18) to classify solar panel operational status into:
    HEALTHY, CRACKED, HOTSPOT, or SHADOW.
    """
    def __init__(
        self, 
        scada_model_path: str = "models/xgboost_scada.joblib", 
        scaler_path: str = "models/scada_scaler.joblib", 
        vision_model_path: str = "models/resnet18_thermal_best.pth",
        label_encoder_path: str = "models/label_encoder.joblib"
    ):
        # 1. Load SCADA Telemetry Model & Scaler
        self.scada_model = joblib.load(scada_model_path) if os.path.exists(scada_model_path) else None
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        self.scada_features = [
            "DC_POWER", "AC_POWER", "AMBIENT_TEMPERATURE", 
            "MODULE_TEMPERATURE", "IRRADIATION"
        ]

        # 2. Load Label Encoder (maps output indices 0, 1, 2, 3 to class names)
        self.label_encoder = joblib.load(label_encoder_path) if os.path.exists(label_encoder_path) else None

        # 3. Load ResNet-18 Thermal Vision Model
        self.vision_model = None
        self.device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

        if vision_model_path and os.path.exists(vision_model_path) and TORCH_AVAILABLE:
            try:
                loaded_obj = torch.load(vision_model_path, map_location=self.device)
                
                if isinstance(loaded_obj, torch.nn.Module):
                    self.vision_model = loaded_obj
                elif isinstance(loaded_obj, dict):
                    # Reconstruct ResNet18 if state_dict was saved
                    num_classes = len(self.label_encoder.classes_) if self.label_encoder else 4
                    model = models.resnet18(weights=None)
                    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                    state_dict = loaded_obj.get("state_dict", loaded_obj)
                    model.load_state_dict(state_dict)
                    self.vision_model = model
                
                if self.vision_model:
                    self.vision_model.to(self.device)
                    self.vision_model.eval()
            except Exception as e:
                print(f"[Warning] Failed to load ResNet18 thermal model: {e}")
                self.vision_model = None

        # Standard ResNet Image Transforms
        if TORCH_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _analyze_scada_xai(self, scada_input: dict) -> dict:
        """
        Calculates expected DC power generation based on irradiance and module temperature,
        computes output drop residuals, and runs XGBoost telemetry prediction.
        """
        dc_power = float(scada_input.get("DC_POWER", 0.0))
        ac_power = float(scada_input.get("AC_POWER", 0.0))
        irradiation = float(scada_input.get("IRRADIATION", 0.0))
        mod_temp = float(scada_input.get("MODULE_TEMPERATURE", 25.0))

        # Theoretical Yield Benchmark
        temp_coefficient = 0.004  # -0.4% per °C above 25°C
        temp_loss = 1.0 - max(0.0, (mod_temp - 25.0) * temp_coefficient)
        expected_dc = irradiation * 200.0 * temp_loss
        
        # Calculate Power Residual Drop (%)
        if expected_dc > 10.0:
            power_deficit_pct = float(max(0.0, (expected_dc - dc_power) / expected_dc * 100))
        else:
            power_deficit_pct = 0.0

        # Inverter AC/DC Conversion Efficiency (%)
        inverter_efficiency = float((ac_power / (dc_power + 1e-5)) * 100) if dc_power > 0 else 0.0

        # XGBoost Model Inference
        scada_fault_prob = 0.10
        if self.scada_model:
            df = pd.DataFrame([scada_input])[self.scada_features]
            scaled_df = self.scaler.transform(df) if self.scaler else df.values
            if hasattr(self.scada_model, "predict_proba"):
                scada_fault_prob = float(self.scada_model.predict_proba(scaled_df)[0, 1])

        # Feature Attribution Diagnostics
        feature_contributions = {}
        if power_deficit_pct > 25.0 and irradiation >= 0.3:
            feature_contributions["DC_POWER"] = f"Unexplained power drop of {power_deficit_pct:.1f}% under normal irradiance ({irradiation} kW/m²)."
        if irradiation < 0.35:
            feature_contributions["IRRADIATION"] = f"Low solar irradiance detected ({irradiation} kW/m²), driving yield drop."
        if mod_temp > 55.0:
            feature_contributions["MODULE_TEMPERATURE"] = f"Elevated thermal reading ({mod_temp}°C) causing thermal efficiency loss."
        if inverter_efficiency < 80.0 and dc_power > 20.0:
            feature_contributions["AC_POWER"] = f"Low inverter efficiency ({inverter_efficiency:.1f}%), suggesting AC side conversion issue."

        return {
            "scada_fault_prob": round(scada_fault_prob, 4),
            "expected_dc_power": round(expected_dc, 2),
            "actual_dc_power": round(dc_power, 2),
            "power_deficit_pct": round(power_deficit_pct, 1),
            "inverter_efficiency_pct": round(inverter_efficiency, 1),
            "key_telemetry_drivers": feature_contributions
        }

    def _analyze_vision_xai(self, image_input) -> dict:
        """
        Executes ResNet-18 Thermal Vision inference and decodes classes via Label Encoder.
        """
        if image_input is None or self.vision_model is None or not TORCH_AVAILABLE:
            return {"primary_detected_class": None, "max_confidence": 0.0, "class_probabilities": {}}

        try:
            # Format input image
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return {"primary_detected_class": None, "max_confidence": 0.0, "class_probabilities": {}}
                img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            else:
                return {"primary_detected_class": None, "max_confidence": 0.0, "class_probabilities": {}}

            # ResNet-18 Preprocessing and Model Pass
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.vision_model(img_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            top_idx = int(np.argmax(probs))
            max_conf = float(probs[top_idx])

            # Decode class name using LabelEncoder
            if self.label_encoder and hasattr(self.label_encoder, "inverse_transform"):
                predicted_class = str(self.label_encoder.inverse_transform([top_idx])[0]).upper()
                class_names = [str(c).upper() for c in self.label_encoder.classes_]
            else:
                default_classes = ["HEALTHY", "CRACKED", "HOTSPOT", "SHADOW"]
                predicted_class = default_classes[top_idx] if top_idx < len(default_classes) else "UNKNOWN"
                class_names = default_classes

            class_probs = {class_names[i]: round(float(probs[i]), 4) for i in range(min(len(class_names), len(probs)))}

            return {
                "primary_detected_class": predicted_class,
                "max_confidence": round(max_conf, 4),
                "class_probabilities": class_probs
            }
        except Exception as e:
            print(f"[Warning] Thermal vision inference failed: {e}")
            return {"primary_detected_class": None, "max_confidence": 0.0, "class_probabilities": {}}

    def evaluate_panel(
        self, 
        scada_input: dict, 
        thermal_image_input=None, 
        simulated_vision_class: str = None
    ) -> dict:
        """
        Multimodal Decision Engine with Explainable AI output.
        Integrates XGBoost SCADA telemetry metrics with ResNet-18 Thermal IR Vision outputs.
        """
        # Step 1: Explainable SCADA Diagnostics
        scada_xai = self._analyze_scada_xai(scada_input)

        # Step 2: Explainable Vision Diagnostics
        vision_xai = self._analyze_vision_xai(thermal_image_input)
        
        # Testing simulation override (if image not provided during quick test)
        if simulated_vision_class:
            vision_xai["primary_detected_class"] = simulated_vision_class.upper()
            vision_xai["max_confidence"] = 0.92

        vis_class = vision_xai.get("primary_detected_class")
        vis_conf = vision_xai.get("max_confidence", 0.0)

        # Step 3: Combined Decision Logic & Diagnosis Categorization
        final_status = "HEALTHY"
        primary_cause = ""
        xai_factors = []

        # Decision Rule 1: CRACKED (Physical surface fracture)
        if vis_class == "CRACKED" and vis_conf >= 0.40:
            final_status = "CRACKED"
            primary_cause = "Physical mechanical crack detected on panel surface via thermal imaging."
            xai_factors.append(f"ResNet-18 thermal vision stream identified structural crack ({vis_conf * 100:.1f}% confidence).")
            if scada_xai["power_deficit_pct"] > 15.0:
                xai_factors.append(f"Active fault is causing a {scada_xai['power_deficit_pct']}% electrical output drop.")

        # Decision Rule 2: HOTSPOT (Cell degradation / localized overheating)
        elif vis_class == "HOTSPOT" or (scada_xai["scada_fault_prob"] > 0.60 and scada_input.get("MODULE_TEMPERATURE", 0) > 55.0):
            final_status = "HOTSPOT"
            primary_cause = "Localized thermal hotspot detected causing thermal resistance and cell stress."
            if vis_class == "HOTSPOT":
                xai_factors.append(f"ResNet-18 thermal vision stream detected hotspot anomaly ({vis_conf * 100:.1f}% confidence).")
            if scada_input.get("MODULE_TEMPERATURE", 0) > 55.0:
                xai_factors.append(f"Elevated module temperature at {scada_input.get('MODULE_TEMPERATURE')}°C exceeds thermal threshold.")

        # Decision Rule 3: SHADOW (Soiling, cloud cover, or temporary obstruction)
        elif vis_class == "SHADOW" or (scada_xai["power_deficit_pct"] > 25.0 and scada_input.get("IRRADIATION", 0) < 0.40):
            final_status = "SHADOW"
            primary_cause = "Transient optical shadow, cloud cover, or surface soiling suppressing light absorption."
            if vis_class == "SHADOW":
                xai_factors.append(f"ResNet-18 thermal vision stream detected shadow/soiling pattern ({vis_conf * 100:.1f}% confidence).")
            if scada_input.get("IRRADIATION", 0) < 0.40:
                xai_factors.append(f"Solar irradiance is suppressed at {scada_input.get('IRRADIATION')} kW/m².")

        # Decision Rule 4: HEALTHY
        else:
            final_status = "HEALTHY"
            primary_cause = "Panel operating within nominal electrical and thermal limits."
            xai_factors.append(f"Actual power ({scada_xai['actual_dc_power']}W) aligns with expected target ({scada_xai['expected_dc_power']}W).")
            xai_factors.append("No critical visual surface anomalies or hotspot defects detected.")

        # Append telemetry drivers into XAI reasoning list
        for key, value in scada_xai["key_telemetry_drivers"].items():
            xai_factors.append(f"Telemetry Driver ({key}): {value}")

        return {
            "prediction": final_status,
            "explainable_ai": {
                "root_cause_summary": primary_cause,
                "contributing_factors": xai_factors,
                "telemetry_metrics": scada_xai,
                "vision_metrics": vision_xai
            }
        }
        

# ---------------------------------------------------------
# BENCHMARKING PREDICTION TIME
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialize the detector model
    detector = ExplainableSolarFaultDetector()

    # 2. Setup sample inputs (1 SCADA dict + 1 test image)
    sample_scada = {
        "DC_POWER": 2500.0,
        "AC_POWER": 2400.0,
        "AMBIENT_TEMPERATURE": 30.0,
        "MODULE_TEMPERATURE": 45.0,
        "IRRADIATION": 0.8
    }

    # Pass a real image path or create a dummy RGB image for timing
    test_image = Image.new("RGB", (224, 224), color=(200, 100, 50))
    # Or use: test_image = "path/to/sample_thermal_image.jpg"

    num_runs = 100  # Number of samples to average over

    print(f"Benchmarking prediction time across {num_runs} runs...")

    # 3. Warm-up pass (forces CUDA memory allocation & initial initialization)
    _ = detector.evaluate_panel(scada_input=sample_scada, thermal_image_input=test_image)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 4. Measure execution time
    start_time = time.perf_counter()

    for _ in range(num_runs):
        _ = detector.evaluate_panel(scada_input=sample_scada, thermal_image_input=test_image)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    # 5. Compute Latency and Throughput
    total_time_sec = end_time - start_time
    avg_latency_ms = (total_time_sec / num_runs) * 1000
    fps = num_runs / total_time_sec

    print(f"\n--- INFERENCE PERFORMANCE METRICS ---")
    print(f"Total Time for {num_runs} predictions: {total_time_sec:.4f} seconds")
    print(f"Average Prediction Time per Sample: {avg_latency_ms:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS (Predictions / Sec)")