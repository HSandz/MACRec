from recbole.quick_start import run_recbole
import torch.distributed as dist
from tools import get_topk_items

## User-defined settings
my_dataset = 'amazon-beauty'
my_model = 'SASRec'
my_config_dict = {}
if my_dataset == 'yelp-2020':
    my_config_dict['val_interval'] = {'timestamp': "[1546272000, 1577808000)"}
if my_model == 'LightGCN':
    my_config_dict['train_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 1, 'dynamic': False}

## Patch for distributed barrier
def barrier_patch(*args, **kwargs):
    if dist.is_initialized():
        return original_barrier(*args, **kwargs)
    else:
        return None
original_barrier = dist.barrier
dist.barrier = barrier_patch

## Run recbole training
run_recbole(
    model = my_model,
    dataset = my_dataset,
    config_file_list=['config.yaml'],
    config_dict = my_config_dict
)

## Get top-k items after training
get_topk_items.main()