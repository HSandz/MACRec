import pandas as pd
import ast
import sys
from collections import Counter

def print_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def parse_candidates(df):
    """Parse candidate lists and check GT inclusion"""
    results = []
    for _, row in df.iterrows():
        cands = ast.literal_eval(row['candidate_item_id'])
        gt = row['item_id']
        gt_in = gt in cands
        pos = cands.index(gt) if gt_in else None
        results.append({'cands': cands, 'gt': gt, 'gt_in': gt_in, 'pos': pos})
    return results

def main():
    # Load data
    test_file = sys.argv[1] if len(sys.argv) > 1 else 'data/ml-100k/test_SGCL.csv'
    df = pd.read_csv(test_file)
    
    # Basic info
    print_section(f"FILE: {test_file}")
    print(f"Total samples: {len(df)}")
    
    # Parse all candidates
    results = parse_candidates(df)
    
    # Candidate counts
    cand_counts = [len(r['cands']) for r in results]
    print(f"\nCandidate count distribution:")
    print(pd.Series(cand_counts).value_counts().sort_index())
    
    # GT inclusion stats
    gt_in_count = sum(r['gt_in'] for r in results)
    gt_rate = gt_in_count / len(results) * 100
    print_section("GROUND TRUTH STATISTICS")
    print(f"GT in candidates: {gt_in_count}/{len(results)} ({gt_rate:.1f}%)")
    
    # Position stats
    positions = [r['pos'] for r in results if r['pos'] is not None]
    if positions:
        print_section("GT POSITION DISTRIBUTION")
        print(f"Average: {sum(positions)/len(positions):.2f}")
        print(f"Range: {min(positions)} - {max(positions)}")
        print(f"\nTop 20 positions:")
        for pos, count in Counter(positions).most_common(20):
            print(f"  Position {pos}: {count} times")

if __name__ == "__main__":
    main()
