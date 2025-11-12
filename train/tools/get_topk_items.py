import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def find_checkpoint_files():
    """Find all .pth checkpoint files in the saved directory."""
    saved_dir = os.path.join(project_root, 'saved')
    if not os.path.exists(saved_dir):
        return []
    
    checkpoint_files = [f for f in os.listdir(saved_dir) if f.endswith('.pth')]
    return sorted(checkpoint_files, reverse=True)  # Newest first

def select_checkpoint():
    """Interactive checkpoint selection."""
    checkpoint_files = find_checkpoint_files()
    
    if not checkpoint_files:
        print("❌ Error: No checkpoint files found in 'saved' directory!")
        return None
    
    print("="*70)
    print("RecBole Top-K Items Generator")
    print("="*70)
    print("\n📁 Available checkpoint files:")
    for i, checkpoint_file in enumerate(checkpoint_files, 1):
        print(f"  {i}. {checkpoint_file}")
    print(f"  {len(checkpoint_files) + 1}. Use latest file (default)")
    
    while True:
        try:
            choice = input(f"\nNhập lựa chọn (Enter choice) [1-{len(checkpoint_files) + 1}] (mặc định: {len(checkpoint_files) + 1}): ").strip()
            
            if choice == '' or choice == str(len(checkpoint_files) + 1):
                # Use latest file (first in sorted list)
                selected_file = checkpoint_files[0]
                break
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(checkpoint_files):
                    selected_file = checkpoint_files[choice_idx]
                    break
                else:
                    print(f"❌ Vui lòng nhập số từ 1 đến {len(checkpoint_files) + 1}")
        except ValueError:
            print("❌ Vui lòng nhập một số hợp lệ")
        except KeyboardInterrupt:
            print("\n\nĐã hủy bỏ.")
            return None
    
    checkpoint_path = os.path.join(project_root, 'saved', selected_file)
    print(f"\n✓ Selected checkpoint: {selected_file}")
    return checkpoint_path

# Select checkpoint file
checkpoint_path = select_checkpoint()
if checkpoint_path is None:
    exit(1)

# Load model and data
print(f"\nLoading model and data from: {checkpoint_path}")
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(checkpoint_path)

device = config['device']  # GPU or CPU
model.eval()

# Display model and dataset info
# RecBole Config object supports dictionary access but not .get() method
try:
    dataset_name = config['dataset']
except (KeyError, AttributeError):
    dataset_name = 'unknown'

try:
    model_name = config['model']
except (KeyError, AttributeError):
    model_name = 'unknown'

print(f"✓ Dataset: {dataset_name}")
print(f"✓ Model: {model_name}")
print(f"✓ Device: {device}")
print("="*70)

user_field = dataset.uid_field
item_field = dataset.iid_field

# Get all user IDs from dataset (internal IDs)
# Get all users from field2id_token, excluding [PAD]
field2id_token = dataset.field2id_token[user_field]
uid_list = []
for internal_id, token in enumerate(field2id_token):
    if token != '[PAD]' and token is not None:
        uid_list.append(internal_id)

print(f"\nTotal users: {len(uid_list)}")
print(f"Getting top 20 items for each user (excluding history items)...")

K = 20
batch_size = 64  # Process users in batches to avoid memory issues
topk_all = []

# Process users in batches
for i in tqdm(range(0, len(uid_list), batch_size), desc="Processing batches"):
    batch_uid_list = uid_list[i:i+batch_size]
    
    # Get top-k items for this batch of users
    # full_sort_topk automatically excludes history items and [pad]
    topk_scores, topk_indices = full_sort_topk(
        uid_series=batch_uid_list,
        model=model,
        test_data=test_data,
        k=K,
        device=device
    )
    
    # Use internal IDs directly
    for j, uid_internal in enumerate(batch_uid_list):
        # Get item internal IDs
        item_internal_ids = topk_indices[j].cpu().numpy()
        
        # Store results using internal IDs
        topk_all.append({
            'user_id': uid_internal,
            **{f'item_{idx+1}': int(item_id) for idx, item_id in enumerate(item_internal_ids)}
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(topk_all)

# Create output directory
path_output = os.path.join(project_root, 'recommendations')
os.makedirs(path_output, exist_ok=True)

# Format filename: {dataset}-for-{model}.csv
output_file = os.path.join(path_output, f'{dataset_name}-for-{model_name}.csv')
df.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ COMPLETED SUCCESSFULLY")
print("="*70)
print(f"✓ File saved at: {output_file}")
print(f"✓ Total users processed: {len(topk_all)}")
print(f"✓ Each user has {K} recommended items (history items excluded)")
print(f"✓ Dataset: {dataset_name}")
print(f"✓ Model: {model_name}")
print("="*70)

