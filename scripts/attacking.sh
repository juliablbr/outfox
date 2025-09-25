cd ../
# OUTFOX attack
python -m src.attacking.outfox_attacking

# DIPPER attack on various models
python -m src.attacking.dipper_attacking --model_name chatgpt
python -m src.attacking.dipper_attacking --model_name flan_t5_xxl
python -m src.attacking.dipper_attacking --model_name text_davinci_003
