# 1. Task: rp
# 1.1 Dataset: ml-100k
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/ml-100k-reflection.jsonl
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/ml-100k-v2.jsonl --model_path ckpts/xxxx/epoch-0
# 1.2 Dataset: beauty
# python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/Beauty-reflection.jsonl
# python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/Beauty-v2.jsonl --model_path ckpts/xxxx/epoch-0

# 2. Task: sr
# 2.1 Dataset: ml-100k
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/ml-100k-reflection.jsonl
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/ml-100k-v1.jsonl --model_path ckpts/xxxx/epoch-0
# 2.2 Dataset: beauty
# python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/Beauty-reflection.jsonl
# python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/Beauty-v1.jsonl --model_path ckpts/xxxx/epoch-0

# Get topk items from retrieved.pkl
python train/tools/get_topk_items_from_pkl.py --pkl_file train\saved\lru\ml-100k\retrieved.pkl
python train/tools/get_topk_items_from_pkl.py --pkl_file train\saved\lru\yelp2020\retrieved.pkl

# Train recbole model
python train/run.py

# Get topk items from recbole_model.pth
python train/tools/get_topk_items.py --pth_file train\saved\recbole\ml-100k\mode.pth
python train/tools/get_topk_items.py --pth_file train\saved\recbole\yelp2020\model.pth
