# import os
# from src.integrated_model import ExplainableSolarFaultDetector

# def run_real_image_tests():
#     # 1. Initialize detector
#     detector = ExplainableSolarFaultDetector(
#         scada_model_path="models/xgboost_scada.joblib",
#         scaler_path="models/scada_scaler.joblib",
#         vision_model_path="models/resnet18_thermal_best.pth",
#         label_encoder_path="models/label_encoder.joblib"
#     )

#     # 2. Define real image paths and matching SCADA telemetry profiles
#     test_cases = [
#         {
#             "scenario": "SCENARIO 1: HEALTHY PANEL",
#             "image_path": "data/healthy_panel.png",
#             "telemetry": {
#                 "DC_POWER": 170.0,
#                 "AC_POWER": 162.0,
#                 "AMBIENT_TEMPERATURE": 28.0,
#                 "MODULE_TEMPERATURE": 35.0,
#                 "IRRADIATION": 0.85
#             }
#         },
#         {
#             "scenario": "SCENARIO 2: CRACKED PANEL",
#             "image_path": "data/cracked_panel.png",
#             "telemetry": {
#                 "DC_POWER": 100.0,
#                 "AC_POWER": 92.0,
#                 "AMBIENT_TEMPERATURE": 30.0,
#                 "MODULE_TEMPERATURE": 42.0,
#                 "IRRADIATION": 0.85
#             }
#         },
#         {
#             "scenario": "SCENARIO 3: HOTSPOT DEFECT",
#             "image_path": "data/hotspot_panel.png",
#             "telemetry": {
#                 "DC_POWER": 110.0,
#                 "AC_POWER": 102.0,
#                 "AMBIENT_TEMPERATURE": 32.0,
#                 "MODULE_TEMPERATURE": 68.0,
#                 "IRRADIATION": 0.85
#             }
#         },
#         {
#             "scenario": "SCENARIO 4: SHADOW / SOILING",
#             "image_path": "data/shading_panel.png",
#             "telemetry": {
#                 "DC_POWER": 35.0,
#                 "AC_POWER": 31.0,
#                 "AMBIENT_TEMPERATURE": 25.0,
#                 "MODULE_TEMPERATURE": 28.0,
#                 "IRRADIATION": 0.20
#             }
#         },
#         {
#             "scenario": "SCENARIO 5: AMBIGUOUS (BORDERLINE DEFECT)",
#             "image_path": "data/ambiguous_defect.png",
#             "telemetry": {
#                 "DC_POWER": 115.0,
#                 "AC_POWER": 108.0,
#                 "AMBIENT_TEMPERATURE": 30.0,
#                 "MODULE_TEMPERATURE": 58.0,
#                 "IRRADIATION": 0.85
#             }
#         }
#     ]

#     print("==================================================")
#     print("      REAL IMAGE & MULTIMODAL INTEGRATION TEST    ")
#     print("==================================================\n")

#     for case in test_cases:
#         print(f"--- Running {case['scenario']} ---")
#         img_path = case["image_path"]

#         if not os.path.exists(img_path):
#             print(f"[SKIP] Image file not found at path: {img_path}")
#             print("Please place a valid image at this path to run live inference.\n" + "-"*50 + "\n")
#             continue

#         # Execute full dual-stream inference
#         result = detector.evaluate_panel(
#             scada_input=case["telemetry"],
#             thermal_image_input=img_path,
#             simulated_vision_class=None  # Full live forward pass through ResNet-18
#         )

#         vision_metrics = result["explainable_ai"]["vision_metrics"]
#         class_probs = vision_metrics.get("class_probabilities", {})

#         print(f"Final Model Prediction : {result['prediction']}")
#         print(f"Vision Classification  : {vision_metrics.get('primary_detected_class')} (Confidence: {vision_metrics.get('max_confidence')})")
#         print("Class Probability Map  :")
#         for label, prob in class_probs.items():
#             print(f"  • {label:<10}: {prob * 100:.2f}%")
        
#         # Check for non-clear winner (close top two probabilities)
#         sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
#         if len(sorted_probs) >= 2:
#             top_class, top_prob = sorted_probs[0]
#             second_class, second_prob = sorted_probs[1]
#             diff = top_prob - second_prob
#             if diff < 0.15:  # Margin smaller than 15%
#                 print(f"\n[AMBIGUITY FLAG] Borderline prediction! {top_class} ({top_prob*100:.1f}%) vs {second_class} ({second_prob*100:.1f}%). Margin: {diff*100:.1f}%")

#         print(f"Root Cause Summary     : {result['explainable_ai']['root_cause_summary']}")
#         print("XAI Drivers            :")
#         for factor in result['explainable_ai']['contributing_factors']:
#             print(f"  • {factor}")
#         print("\n" + "-"*50 + "\n")

# if __name__ == "__main__":
#     run_real_image_tests()


from src.integrated_model import ExplainableSolarFaultDetector

detector = ExplainableSolarFaultDetector()

# Simulate a visually ambiguous panel (High temp + active crack)
ambiguous_telemetry = {
    "DC_POWER": 115.0, "AC_POWER": 108.0,
    "AMBIENT_TEMPERATURE": 30.0, "MODULE_TEMPERATURE": 58.0,
    "IRRADIATION": 0.85
}

# Evaluate with simulated split confidence
result = detector.evaluate_panel(
    scada_input=ambiguous_telemetry,
    thermal_image_input="data/ambiguous_defect.jpg",
    simulated_vision_class="HOTSPOT" # Overrides primary class to test decision thresholding
)

print("Status Prediction :", result["prediction"])
print("XAI Drivers       :", result["explainable_ai"]["contributing_factors"])