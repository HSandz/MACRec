my_dataset = 'amazon-beauty'
my_model = 'SASRec'

import torch.distributed as dist
original_barrier = dist.barrier

def barrier_patch(*args, **kwargs):
    if dist.is_initialized():
        return original_barrier(*args, **kwargs)
    else:
        return None

dist.barrier = barrier_patch

from recbole.quick_start import run_recbole

my_config_dict = {}
if my_dataset == 'yelp-2020':
    my_config_dict['val_interval'] = {'timestamp': "[1546272000, 1577808000)"}
if my_model == 'LightGCN':
    my_config_dict['train_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 1, 'dynamic': False}

run_recbole(
    model = my_model,
    dataset = my_dataset,
    config_file_list=['config.yaml'],
    config_dict = my_config_dict
)

from tools.get_topk_items import get_topk_items
get_topk_items()