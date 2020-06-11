cd 1_evaluate_filter_importance
python proxy_dataset.py --yaml_file ../config.yaml
python generate_mask.py --yaml_file ../config.yaml 2>&1 | tee "log_mask.txt"
python generate_index.py  # you should try this script several times (try different top_k value) to find the compress rate
