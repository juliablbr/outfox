cd ../
# OUTFOX detection without considering attacks on test set by various LMs
python -m src.detection.outfox_detection_without_considering_attack --model_name chatgpt
python -m src.detection.outfox_detection_without_considering_attack --model_name flan_t5_xxl
python -m src.detection.outfox_detection_without_considering_attack --model_name text_davinci_003

# OUTFOX detection with considering attacks
python -m src.detection.outfox_detection_with_considering_attack --attacking_method outfox --model_name chatgpt
python -m src.detection.outfox_detection_with_considering_attack --attacking_method outfox --model_name flan_t5_xxl
python -m src.detection.outfox_detection_with_considering_attack --attacking_method outfox --model_name text_davinci_003

python -m src.detection.outfox_detection_with_considering_attack --attacking_method paraphrase --model_name chatgpt
python -m src.detection.outfox_detection_with_considering_attack --attacking_method paraphrase --model_name flan_t5_xxl
python -m src.detection.outfox_detection_with_considering_attack --attacking_method paraphrase --model_name text_davinci_003

# Baseline detections on a dev set
python -m src.detection.baseline_detection_dev --base_model_name chatgpt
python -m src.detection.baseline_detection_dev --base_model_name flan_t5_xxl
python -m src.detection.baseline_detection_dev --base_model_name text_davinci_003

# Baseline detections on a test set
python -m src.detection.baseline_detection_test --base_model_name chatgpt
python -m src.detection.baseline_detection_test --base_model_name flan_t5_xxl
python -m src.detection.baseline_detection_test --base_model_name text_davinci_003
