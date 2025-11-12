from tqdm import tqdm
import pandas as pd
import os
import torch
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model

import torch.distributed as dist
original_barrier = dist.barrier
def barrier_patch(*args, **kwargs):
    if dist.is_initialized():
        return original_barrier(*args, **kwargs)
    else:
        return None
dist.barrier = barrier_patch

original_torch_load = torch.load
def torch_load_patch(*args, **kwargs):
    if 'map_location' not in kwargs and not torch.cuda.is_available():
        kwargs['map_location'] = torch.device('cpu')
    return original_torch_load(*args, **kwargs)
torch.load = torch_load_patch

script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.dirname(script_dir)

def find_checkpoint_files():
    """Find all .pth checkpoint files in the saved directory."""
    saved_dir = os.path.join(train_dir, 'saved')
    if not os.path.exists(saved_dir):
        return []
    
    checkpoint_files = [f for f in os.listdir(saved_dir) if f.endswith('.pth')]
    return sorted(checkpoint_files, reverse=True)

def select_checkpoint():
    """Interactive checkpoint selection."""
    checkpoint_files = find_checkpoint_files()
    
    if not checkpoint_files:
        print("Error: No checkpoint files found in 'saved' directory!")
        return None
    
    print("Available checkpoint files:")
    for i, checkpoint_file in enumerate(checkpoint_files, 1):
        print(f"  {i}. {checkpoint_file}")
    print(f"  {len(checkpoint_files) + 1}. Use latest file (default)")
    
    while True:
        try:
            choice = input(f"\nEnter choice [1-{len(checkpoint_files) + 1}] (default: {len(checkpoint_files) + 1}): ").strip()
            
            if choice == '' or choice == str(len(checkpoint_files) + 1):
                selected_file = checkpoint_files[0]
                break
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(checkpoint_files):
                    selected_file = checkpoint_files[choice_idx]
                    break
                else:
                    print(f"Please enter a number from 1 to {len(checkpoint_files) + 1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None
    
    checkpoint_path = os.path.join(train_dir, 'saved', selected_file)
    return checkpoint_path

checkpoint_path = select_checkpoint()
if checkpoint_path is None:
    exit(1)

if not torch.cuda.is_available():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint['config']['device'] = 'cpu'
    temp_checkpoint_path = checkpoint_path + '.cpu_temp'
    torch.save(checkpoint, temp_checkpoint_path)
    config, model, dataset, _, _, test_data = load_data_and_model(temp_checkpoint_path)
    if os.path.exists(temp_checkpoint_path):
        os.remove(temp_checkpoint_path)
else:
    config, model, dataset, _, _, test_data = load_data_and_model(checkpoint_path)

device = config['device']
model.eval()

try:
    dataset_name = config['dataset']
except (KeyError, AttributeError):
    dataset_name = 'unknown'

try:
    model_name = config['model']
except (KeyError, AttributeError):
    model_name = 'unknown'

user_field = dataset.uid_field
item_field = dataset.iid_field

field2id_token = dataset.field2id_token[user_field]
uid_list = []
for internal_id, token in enumerate(field2id_token):
    if token != '[PAD]' and token is not None:
        uid_list.append(internal_id)

gt_items_dict = {}

for batch in test_data:
    if isinstance(batch, tuple):
        interaction = batch[0]
    else:
        interaction = batch
    
    users = interaction[user_field]
    items = interaction[item_field]
    
    for uid, iid in zip(users.tolist(), items.tolist()):
        if uid not in gt_items_dict:
            gt_items_dict[uid] = iid

print(f"Loaded {len(gt_items_dict)} ground truth items from test_data")

K = 20
batch_size = 64
topk_all = []

for i in tqdm(range(0, len(uid_list), batch_size), desc="Processing batches"):
    batch_uid_list = uid_list[i:i+batch_size]
    
    _, topk_indices = full_sort_topk(
        uid_series=batch_uid_list,
        model=model,
        test_data=test_data,
        k=K,
        device=device
    )
    
    for j, uid_internal in enumerate(batch_uid_list):
        item_internal_ids = topk_indices[j].cpu().numpy()
        gt_item = gt_items_dict.get(uid_internal, None)
        
        topk_all.append({
            'user_id': uid_internal,
            'gt_item': int(gt_item) if gt_item is not None else '',
            **{f'item_{idx+1}': int(item_id) for idx, item_id in enumerate(item_internal_ids)}
        })

df = pd.DataFrame(topk_all)

path_output = os.path.join(train_dir, 'recommendations')
os.makedirs(path_output, exist_ok=True)

output_file = os.path.join(path_output, f'{dataset_name}-for-{model_name}.csv')
df.to_csv(output_file, index=False)

print(f"File saved at: {output_file}")
print(f"Total users processed: {len(topk_all)}")