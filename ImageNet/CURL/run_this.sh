# 1. calculate the importance score for each filter
cd 1_evaluate_filter_importance
python proxy_dataset.py
python generate_mask.py 2>&1 | tee "log_mask.txt"
python generate_index.py  # you should try this script several times (try different top_k value) to find the compress rate

# 2. fine-tuned the pruned model
cd ../2_fine_tuning
python main.py --gpu_id 0,1,2,3 2>&1 | tee "log.txt"
