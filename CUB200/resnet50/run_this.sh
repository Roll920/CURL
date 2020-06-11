config_file='../config.yaml'

cd 2_mixup_kd
python main.py --yaml_file ${config_file} 2>&1 | tee "log.txt"

cd ../3_label_refine
python extract_logits.py --yaml_file ${config_file} 2>&1 | tee "log_logits.txt"
python main.py --yaml_file ${config_file} 2>&1 | tee "log.txt"
