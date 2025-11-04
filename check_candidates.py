import pandas as pd
import ast
import sys

# Get test file from command line or use default
test_file = sys.argv[1] if len(sys.argv) > 1 else 'data/ml-100k/test_SGCL.csv'

# Read the test file
df = pd.read_csv(test_file)

# Count candidates per row
df['candidate_count'] = df['candidate_item_id'].apply(lambda x: len(ast.literal_eval(x)))

print("="*60)
print(f"FILE: {test_file}")
print("="*60)
print(f"\nTotal test samples: {len(df)}")
print(f"\nCandidate item counts distribution:")
print(df['candidate_count'].value_counts().sort_index())

expected_count = df['candidate_count'].mode()[0]
print(f"\nAll rows have exactly {expected_count} candidates: {(df['candidate_count'] == expected_count).all()}")

# Check if GT item is in candidates
print("\n" + "="*60)
print("GROUND TRUTH ITEM INCLUSION CHECK")
print("="*60)

gt_in_candidates_count = 0
gt_not_in_candidates_count = 0

for i in range(min(10, len(df))):
    cands = ast.literal_eval(df.iloc[i]['candidate_item_id'])
    gt = df.iloc[i]['item_id']
    is_in = gt in cands
    
    if is_in:
        gt_in_candidates_count += 1
    else:
        gt_not_in_candidates_count += 1
    
    print(f"Sample {i+1}: user={df.iloc[i]['user_id']}, GT={gt}, in_candidates={is_in}")

print(f"\nIn first 10 samples:")
print(f"  GT in candidates: {gt_in_candidates_count}")
print(f"  GT NOT in candidates: {gt_not_in_candidates_count}")

# Check all samples
all_gt_in_candidates = []
for i in range(len(df)):
    cands = ast.literal_eval(df.iloc[i]['candidate_item_id'])
    gt = df.iloc[i]['item_id']
    all_gt_in_candidates.append(gt in cands)

print(f"\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)
print(f"Total samples where GT in candidates: {sum(all_gt_in_candidates)}")
print(f"Total samples where GT NOT in candidates: {len(all_gt_in_candidates) - sum(all_gt_in_candidates)}")
print(f"Percentage GT in candidates: {sum(all_gt_in_candidates)/len(all_gt_in_candidates)*100:.1f}%")

# Show position distribution of GT in candidate list
print(f"\n" + "="*60)
print("GT ITEM POSITION IN CANDIDATE LIST")
print("="*60)
gt_positions = []
for i in range(len(df)):
    cands = ast.literal_eval(df.iloc[i]['candidate_item_id'])
    gt = df.iloc[i]['item_id']
    if gt in cands:
        gt_positions.append(cands.index(gt))

if gt_positions:
    print(f"Average position: {sum(gt_positions)/len(gt_positions):.2f}")
    print(f"Min position: {min(gt_positions)} (first)")
    print(f"Max position: {max(gt_positions)} (last)")
    print(f"\nPosition distribution (first 20 positions):")
    from collections import Counter
    pos_counts = Counter(gt_positions)
    for pos in sorted(pos_counts.keys())[:20]:
        print(f"  Position {pos}: {pos_counts[pos]} times")
