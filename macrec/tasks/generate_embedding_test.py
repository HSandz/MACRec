from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from macrec.tasks.base import Task
from macrec.utils import init_all_seeds


class GenerateEmbeddingTestTask(Task):
    """Generate test CSV files with candidates selected using model embeddings.
    
    This task reads user/item embeddings from a model directory and creates test files
    where candidate items (including target) are the top-K most similar items based on
    cosine similarity of embeddings.
    """
    
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--data_dir', type=str, required=True, 
                          help='Dataset directory (e.g., data/ml-100k)')
        parser.add_argument('--model_dir', type=str, required=True,
                          help='Model embeddings directory (e.g., models/SGCL/ml-100k)')
        parser.add_argument('--model_name', type=str, required=True,
                          help='Model name for output file (e.g., SGCL, LightGCN)')
        parser.add_argument('--n_candidates', type=int, default=20,
                          help='Number of candidate items including target (default: 20)')
        parser.add_argument('--test_file', type=str, default='test.csv',
                          help='Input test file name (default: test.csv)')
        parser.add_argument('--seed', type=int, default=2024,
                          help='Random seed for reproducibility (default: 2024)')
        return parser

    def run(self, data_dir: str, model_dir: str, model_name: str, 
            n_candidates: int = 20, test_file: str = 'test.csv', seed: int = 2024):
        """Generate embedding-based test file.
        
        Args:
            data_dir: Path to dataset directory
            model_dir: Path to model embeddings directory
            model_name: Name of the model (used in output filename)
            n_candidates: Number of candidates to select (including target)
            test_file: Name of input test file
            seed: Random seed
        """
        init_all_seeds(seed)
        
        # Paths
        test_path = os.path.join(data_dir, test_file)
        user_emb_path = os.path.join(model_dir, 'user_embeddings.csv')
        item_emb_path = os.path.join(model_dir, 'item_embeddings.csv')
        output_path = os.path.join(data_dir, f'test_{model_name}.csv')
        
        # Validate inputs
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        if not os.path.exists(user_emb_path):
            raise FileNotFoundError(f"User embeddings not found: {user_emb_path}")
        if not os.path.exists(item_emb_path):
            raise FileNotFoundError(f"Item embeddings not found: {item_emb_path}")
        
        logger.info(f"Loading test data from: {test_path}")
        df_test = pd.read_csv(test_path)
        logger.info(f"Loaded {len(df_test)} test samples")
        
        logger.info(f"Loading user embeddings from: {user_emb_path}")
        df_user_emb = pd.read_csv(user_emb_path)
        user_embeddings = df_user_emb.set_index('user_id').values
        user_ids = df_user_emb['user_id'].values
        logger.info(f"Loaded embeddings for {len(user_ids)} users")
        
        logger.info(f"Loading item embeddings from: {item_emb_path}")
        df_item_emb = pd.read_csv(item_emb_path)
        item_embeddings = df_item_emb.set_index('item_id').values
        item_ids = df_item_emb['item_id'].values
        logger.info(f"Loaded embeddings for {len(item_ids)} items")
        
        # Create mappings for fast lookup
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        
        logger.info(f"Computing item similarities and generating candidates...")
        new_rows = []
        
        for idx, row in df_test.iterrows():
            user_id = row['user_id']
            target_item_id = row['item_id']
            
            # Parse history items to exclude from candidates
            history_str = row['history_item_id']
            if isinstance(history_str, str):
                # Remove brackets and split
                history_items = set(map(int, history_str.strip('[]').split(', ')))
            else:
                history_items = set()
            
            # Check if user and item exist in embeddings
            if user_id not in user_id_to_idx:
                logger.warning(f"User {user_id} not found in embeddings, skipping")
                continue
            if target_item_id not in item_id_to_idx:
                logger.warning(f"Item {target_item_id} not found in embeddings, skipping")
                continue
            
            # Get user embedding
            user_emb = user_embeddings[user_id_to_idx[user_id]].reshape(1, -1)
            
            # Compute similarities with all items
            similarities = cosine_similarity(user_emb, item_embeddings)[0]
            
            # Create candidate pool: exclude ONLY history items (NOT the target)
            # The target may or may not appear in the top-K candidates based on embeddings
            candidate_mask = np.ones(len(item_ids), dtype=bool)
            
            for hist_item in history_items:
                if hist_item in item_id_to_idx:
                    candidate_mask[item_id_to_idx[hist_item]] = False  # Exclude history
            
            # Get top n_candidates most similar items based on embeddings
            candidate_similarities = similarities.copy()
            candidate_similarities[~candidate_mask] = -np.inf  # Mask out excluded items
            
            top_indices = np.argsort(candidate_similarities)[::-1][:n_candidates]
            candidate_items = item_ids[top_indices].tolist()
            
            # All candidates are negatives except if GT happens to be in top-K
            negative_item_ids = [item for item in candidate_items if item != target_item_id]
            
            # Create new row with updated candidates
            new_row = row.copy()
            new_row['neg_item_id'] = str(negative_item_ids)  # Store as string list
            new_row['candidate_item_id'] = str(candidate_items)  # Store as string list
            new_rows.append(new_row)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df_test)} samples")
        
        # Create new dataframe and save
        df_new_test = pd.DataFrame(new_rows)
        df_new_test.to_csv(output_path, index=False)
        
        logger.success(f"Generated {len(df_new_test)} test samples")
        logger.success(f"Output saved to: {output_path}")
        
        # Check how many samples have GT in candidates
        gt_in_candidates_count = 0
        for i in range(len(df_new_test)):
            cands = eval(df_new_test.iloc[i]['candidate_item_id'])
            gt = df_new_test.iloc[i]['item_id']
            if gt in cands:
                gt_in_candidates_count += 1
        
        logger.info(f"Candidates per sample: {n_candidates} (purely from embeddings)")
        logger.info(f"GT item in candidates: {gt_in_candidates_count}/{len(df_new_test)} ({gt_in_candidates_count/len(df_new_test)*100:.1f}%)")
        
        # Print sample
        if len(df_new_test) > 0:
            logger.info("\n=== Sample Output ===")
            sample = df_new_test.iloc[0]
            cands = eval(sample['candidate_item_id'])
            gt = sample['item_id']
            logger.info(f"User: {sample['user_id']}, Target: {gt}")
            logger.info(f"Candidates: {sample['candidate_item_id']}")
            logger.info(f"GT in candidates: {gt in cands}")
            logger.info(f"Negatives: {sample['neg_item_id']}")


if __name__ == '__main__':
    GenerateEmbeddingTestTask().launch()
