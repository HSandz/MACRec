import tensorflow as tf
import numpy as np
import json
import os
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.utils.notebook_utils import store_metadata

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(BASE_DIR, "config.yaml")
user_embedding_file = os.path.join(BASE_DIR, "output/user_embeddings.csv")
item_embedding_file = os.path.join(BASE_DIR, "output/item_embeddings.csv")
user_mapping_file = os.path.join(BASE_DIR, "output/user_id_mapping.json")
item_mapping_file = os.path.join(BASE_DIR, "output/item_id_mapping.json")

# Data Preparation
df = movielens.load_pandas_df(size='100k')
train, test = python_stratified_split(df, ratio=0.75)
data = ImplicitCF(train=train, test=test, seed=42)

# Model Configuration
hparams = prepare_hparams(config_file)
model = LightGCN(hparams, data, seed=42)

# Model Training
with Timer() as train_time:
    model.fit()
print("Took {} seconds for training.".format(train_time.interval))

# Model Evaluation
topk_scores = model.recommend_k_items(test, top_k=10, remove_seen=True)
eval_map = map(test, topk_scores, k=10)
eval_ndcg = ndcg_at_k(test, topk_scores, k=10)
eval_precision = precision_at_k(test, topk_scores, k=10)
eval_recall = recall_at_k(test, topk_scores, k=10)

# Record results for tests - ignore this cell
store_metadata("map", eval_map)
store_metadata("ndcg", eval_ndcg)
store_metadata("precision", eval_precision)
store_metadata("recall", eval_recall)

# Save embeddings
model.infer_embedding(user_embedding_file, item_embedding_file)

# Save mapping files
unique_users = sorted(df['userID'].unique())
unique_items = sorted(df['itemID'].unique())

num_users = len(unique_users)
num_items = len(unique_items)

user_inner_ids = np.arange(num_users)
item_inner_ids = np.arange(num_items)

user_tokens = [str(unique_users[i]) for i in user_inner_ids]
item_tokens = [str(unique_items[i]) for i in item_inner_ids]

with open(user_mapping_file, 'w', encoding='utf-8') as f:
    json.dump({"inner_id": user_inner_ids.tolist(), "token": user_tokens}, f, indent=2)

with open("lightgcn/output/item_id_mapping.json", 'w', encoding='utf-8') as f:
    json.dump({"inner_id": item_inner_ids.tolist(), "token": item_tokens}, f, indent=2)