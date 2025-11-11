from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from macrec.tasks.base import Task
from macrec.utils import init_all_seeds


class SimilarityMethod:
    """Enum for similarity computation methods."""
    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"


class GenerateEmbeddingTestTask(Task):
    """Generate test CSV files with candidates selected using model embeddings.
    
    Supports two types of models:
    1. Collaborative Filtering (CF) models: Use user_embeddings.csv + item_embeddings.csv
       - Examples: LightGCN, SGCL, NCF
       - User representation: Direct user embedding
    
    2. Sequential Recommendation models: Use only item_embeddings.csv
       - Examples: SASRec, GRU4Rec, Caser
       - User representation: Mean of history item embeddings
    
    Creates test files where candidates are top-K most similar items based on
    embedding similarity (purely from embeddings, GT may not be included).
    
    Supports two similarity methods:
    1. Cosine similarity: Measures angle between vectors (normalized, scale-invariant)
    2. Inner product: Raw dot product (sensitive to embedding magnitude)
    
    RecBole Integration:
    Automatically detects and converts RecBole embeddings (which use original MovieLens IDs)
    to match preprocessed IDs used in test.csv. Conversion is done in-place.
    """
    
    @staticmethod
    def _detect_and_convert_recbole_embeddings(model_dir: str, data_dir: str):
        """Detect RecBole embeddings and convert them to use preprocessed IDs.
        
        RecBole uses original dataset IDs (e.g., 1-1682 for MovieLens after filtering),
        but MACRec's preprocessing remaps them to sequential IDs (1-N). This method
        automatically detects this mismatch and converts embeddings in-place.
        
        Args:
            model_dir: Directory containing embeddings (e.g., models/LightGCN/ml-100k/)
            data_dir: Dataset directory containing ID mappings (e.g., data/ml-100k/)
        """
        item_emb_path = os.path.join(model_dir, 'item_embeddings.csv')
        user_emb_path = os.path.join(model_dir, 'user_embeddings.csv')
        
        if not os.path.exists(item_emb_path):
            return  # No embeddings to convert
        
        # Check for ID mapping files
        item_mapping_path = os.path.join(data_dir, 'item_id_mapping.csv')
        user_mapping_path = os.path.join(data_dir, 'user_id_mapping.csv')
        
        if not os.path.exists(item_mapping_path):
            logger.debug(f"No item ID mapping found at {item_mapping_path}, skipping conversion")
            return
        
        # Load item embeddings to check if conversion is needed
        df_item_emb = pd.read_csv(item_emb_path)
        
        # Check for [PAD] token (indicates RecBole format)
        has_pad = False
        if len(df_item_emb) > 0 and df_item_emb.iloc[0]['item_id'] == '[PAD]':
            has_pad = True
            logger.info("Detected [PAD] token in embeddings (RecBole format)")
        
        # Load mapping
        df_item_mapping = pd.read_csv(item_mapping_path)
        # Create mapping with string keys for compatibility with both int and string IDs
        id_map = dict(zip(df_item_mapping['original_id'].astype(str), df_item_mapping['preprocessed_id']))
        
        # Check if conversion is needed by comparing ID ranges
        # If embeddings already use preprocessed IDs (1-N), no conversion needed
        if not has_pad:
            # Check if item IDs in embeddings match preprocessed range
            emb_ids = df_item_emb['item_id'].values
            # Try to check if IDs are numeric
            try:
                emb_ids_numeric = pd.to_numeric(emb_ids, errors='coerce')
                if not pd.isna(emb_ids_numeric).all():
                    emb_min = int(np.nanmin(emb_ids_numeric))
                    emb_max = int(np.nanmax(emb_ids_numeric))
                    expected_max = len(id_map)
                    
                    # If IDs are already in preprocessed range (1-N), skip conversion
                    if emb_min == 1 and emb_max == expected_max:
                        logger.debug(f"Embeddings already use preprocessed IDs (1-{expected_max}), skipping conversion")
                        return
            except (ValueError, TypeError):
                # IDs are not numeric (e.g., Amazon ASINs), will need conversion
                pass
        
        logger.info("=" * 80)
        logger.info("Converting RecBole embeddings to preprocessed IDs")
        logger.info("=" * 80)
        
        # Convert item embeddings
        logger.info(f"Converting item embeddings: {item_emb_path}")
        
        # Remove [PAD] token if present
        if has_pad:
            logger.info("Removing [PAD] token row")
            df_item_emb = df_item_emb.iloc[1:].reset_index(drop=True)
        
        # Convert IDs - handle both integer and string IDs (for different datasets)
        # Keep original IDs as strings if they are strings (e.g., Amazon ASINs)
        original_ids = df_item_emb['item_id'].astype(str)
        df_item_emb['item_id'] = original_ids.map(id_map)
        
        # Check for unmapped IDs
        unmapped = df_item_emb['item_id'].isna()
        if unmapped.any():
            logger.warning(f"Found {unmapped.sum()} unmapped items, removing them")
            logger.warning(f"  Original IDs: {original_ids[unmapped].tolist()[:10]}")
            df_item_emb = df_item_emb[~unmapped].reset_index(drop=True)
        
        # Sort by preprocessed ID
        df_item_emb = df_item_emb.sort_values('item_id').reset_index(drop=True)
        
        # Save converted embeddings (overwrite original)
        logger.info(f"Saving converted embeddings: {len(df_item_emb)} items")
        df_item_emb.to_csv(item_emb_path, index=False)
        logger.success(f"✓ Item embeddings converted successfully")
        
        # Convert user embeddings if they exist
        if os.path.exists(user_emb_path) and os.path.exists(user_mapping_path):
            logger.info(f"Converting user embeddings: {user_emb_path}")
            
            df_user_emb = pd.read_csv(user_emb_path)
            df_user_mapping = pd.read_csv(user_mapping_path)
            # Create mapping with string keys for compatibility with both int and string IDs
            user_id_map = dict(zip(df_user_mapping['original_id'].astype(str), df_user_mapping['preprocessed_id']))
            
            # Remove [PAD] if present
            if len(df_user_emb) > 0 and df_user_emb.iloc[0]['user_id'] == '[PAD]':
                logger.info("Removing [PAD] token row from user embeddings")
                df_user_emb = df_user_emb.iloc[1:].reset_index(drop=True)
            
            # Convert IDs - handle both integer and string IDs
            original_user_ids = df_user_emb['user_id'].astype(str)
            df_user_emb['user_id'] = original_user_ids.map(user_id_map)
            
            # Check for unmapped IDs
            unmapped_users = df_user_emb['user_id'].isna()
            if unmapped_users.any():
                logger.warning(f"Found {unmapped_users.sum()} unmapped users, removing them")
                df_user_emb = df_user_emb[~unmapped_users].reset_index(drop=True)
            
            # Sort by preprocessed ID
            df_user_emb = df_user_emb.sort_values('user_id').reset_index(drop=True)
            
            # Save converted embeddings (overwrite original)
            logger.info(f"Saving converted user embeddings: {len(df_user_emb)} users")
            df_user_emb.to_csv(user_emb_path, index=False)
            logger.success(f"✓ User embeddings converted successfully")
        
        logger.info("=" * 80)
        logger.success("RecBole embedding conversion complete!")
        logger.info("=" * 80)
    
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
        parser.add_argument('--similarity_method', type=str, default='cosine',
                          choices=['cosine', 'inner_product'],
                          help='Similarity computation method (default: cosine). '
                               'Use "cosine" for normalized angle-based similarity or '
                               '"inner_product" for raw dot product.')
        parser.add_argument('--use_pkl', action='store_true',
                          help='Use .pkl file with precomputed scores instead of embeddings')
        parser.add_argument('--pkl_file', type=str, default='retrieved.pkl',
                          help='PKL file name (default: retrieved.pkl)')
        parser.add_argument('--seed', type=int, default=2024,
                          help='Random seed for reproducibility (default: 2024)')
        return parser

    def run(self, data_dir: str, model_dir: str, model_name: str, 
            n_candidates: int = 20, test_file: str = 'test.csv', 
            similarity_method: str = 'cosine', use_pkl: bool = False,
            pkl_file: str = 'retrieved.pkl', seed: int = 2024):
        """Generate embedding-based test file.
        
        Args:
            data_dir: Path to dataset directory
            model_dir: Path to model embeddings directory
            model_name: Name of the model (used in output filename)
            n_candidates: Number of candidates to select (including target)
            test_file: Name of input test file
            similarity_method: Similarity computation method ('cosine' or 'inner_product')
            use_pkl: Whether to use .pkl file with precomputed scores
            pkl_file: Name of pkl file (if use_pkl=True)
            seed: Random seed
        """
        init_all_seeds(seed)
        
        # If using pkl file, delegate to pkl handler
        if use_pkl:
            return self._run_from_pkl(data_dir, model_dir, model_name, n_candidates, 
                                     test_file, pkl_file, seed)
        
        # Paths
        test_path = os.path.join(data_dir, test_file)
        user_emb_path = os.path.join(model_dir, 'user_embeddings.csv')
        item_emb_path = os.path.join(model_dir, 'item_embeddings.csv')
        output_path = os.path.join(data_dir, f'test_{model_name}.csv')
        
        # Validate inputs
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        if not os.path.exists(item_emb_path):
            raise FileNotFoundError(f"Item embeddings not found: {item_emb_path}")
        
        # Automatically detect and convert RecBole embeddings if needed
        self._detect_and_convert_recbole_embeddings(model_dir, data_dir)
        
        logger.info(f"Loading test data from: {test_path}")
        df_test = pd.read_csv(test_path)
        logger.info(f"Loaded {len(df_test)} test samples")
        
        # Check if user embeddings exist (for CF models) or only item embeddings (for sequential models)
        has_user_embeddings = os.path.exists(user_emb_path)
        
        if has_user_embeddings:
            logger.info(f"Loading user embeddings from: {user_emb_path}")
            df_user_emb = pd.read_csv(user_emb_path)
            user_embeddings = df_user_emb.set_index('user_id').values
            user_ids = df_user_emb['user_id'].values
            logger.info(f"Loaded embeddings for {len(user_ids)} users")
            user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
            model_type = "collaborative filtering"
        else:
            logger.info("No user embeddings found - using sequential recommendation mode")
            logger.info("Will compute user representations from history item embeddings")
            user_embeddings = None
            user_id_to_idx = None
            model_type = "sequential recommendation"
        
        logger.info(f"Loading item embeddings from: {item_emb_path}")
        df_item_emb = pd.read_csv(item_emb_path)
        item_embeddings = df_item_emb.set_index('item_id').values
        item_ids = df_item_emb['item_id'].values
        logger.info(f"Loaded embeddings for {len(item_ids)} items")
        logger.info(f"Model type detected: {model_type}")
        
        # Validate similarity method
        if similarity_method not in [SimilarityMethod.COSINE, SimilarityMethod.INNER_PRODUCT]:
            raise ValueError(f"Unknown similarity method: {similarity_method}. "
                           f"Choose from {[SimilarityMethod.COSINE, SimilarityMethod.INNER_PRODUCT]}")
        logger.info(f"Similarity method: {similarity_method}")
        
        # Create mappings for fast lookup
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
                history_items = list(map(int, history_str.strip('[]').split(', ')))
                history_items_set = set(history_items)
            else:
                history_items = []
                history_items_set = set()
            
            # Check if target item exists in embeddings
            if target_item_id not in item_id_to_idx:
                logger.warning(f"Item {target_item_id} not found in embeddings, skipping")
                continue
            
            # Get user representation based on model type
            if has_user_embeddings:
                # Collaborative filtering model: use user embedding
                if user_id not in user_id_to_idx:
                    logger.warning(f"User {user_id} not found in embeddings, skipping")
                    continue
                user_emb = user_embeddings[user_id_to_idx[user_id]].reshape(1, -1)
            else:
                # Sequential model: aggregate history item embeddings
                # Filter history items that exist in embeddings
                valid_history = [h for h in history_items if h in item_id_to_idx]
                
                if len(valid_history) == 0:
                    logger.warning(f"No valid history items for user {user_id}, skipping")
                    continue
                
                # Get embeddings for history items and compute mean as user representation
                history_embs = np.array([item_embeddings[item_id_to_idx[h]] for h in valid_history])
                user_emb = np.mean(history_embs, axis=0).reshape(1, -1)
            
            # Compute similarities with all items using the specified method
            if similarity_method == SimilarityMethod.COSINE:
                similarities = cosine_similarity(user_emb, item_embeddings)[0]
            elif similarity_method == SimilarityMethod.INNER_PRODUCT:
                similarities = np.dot(user_emb, item_embeddings.T)[0]
            else:
                raise ValueError(f"Unknown similarity method: {similarity_method}")
            
            # Create candidate pool: exclude ONLY history items (NOT the target)
            # The target may or may not appear in the top-K candidates based on embeddings
            candidate_mask = np.ones(len(item_ids), dtype=bool)
            
            for hist_item in history_items_set:
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
            new_row = {
                'user_id': row['user_id'],
                'item_id': row['item_id'],
                'rating': row['rating'],
                'history_item_id': row['history_item_id'],
                'history_rating': row['history_rating'],
                'candidate_item_id': str(candidate_items)  # Only update this column
            }
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

    def _run_from_pkl(self, data_dir: str, model_dir: str, model_name: str,
                      n_candidates: int, test_file: str, pkl_file: str, seed: int):
        """Generate test file from precomputed data in .pkl file.

        This matches pkl_to_csv.py behavior where candidates are taken directly from the model's
        predictions, which may include items from the user's history.
        
        Args:
            data_dir: Path to dataset directory
            model_dir: Path to model directory containing .pkl file
            model_name: Name of the model (used in output filename)
            n_candidates: Number of candidates to select (truncates test_topk if needed)
            test_file: Name of input test file
            pkl_file: Name of pkl file containing test_topk/test_probs and test_labels
            seed: Random seed
        """
        init_all_seeds(seed)
        
        # Paths
        test_path = os.path.join(data_dir, test_file)
        pkl_path = os.path.join(model_dir, pkl_file)
        output_path = os.path.join(data_dir, f'test_{model_name}.csv')
        
        # Validate inputs
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")
        
        logger.info(f"Loading test data from: {test_path}")
        df_test = pd.read_csv(test_path)
        logger.info(f"Loaded {len(df_test)} test samples")
        
        # IMPORTANT: Sort by user_id to match PKL order (PKL is generated with users in sorted order)
        logger.info("Sorting test data by user_id to match PKL order...")
        df_test = df_test.sort_values('user_id').reset_index(drop=True)
        logger.info(f"First 5 users after sorting: {df_test['user_id'].head(5).tolist()}")
        
        logger.info(f"Loading precomputed data from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        logger.info(f"PKL file keys: {list(pkl_data.keys())}")
        
        # ---- Find top-K candidates or compute from scores (EXACT same logic as pkl_to_csv.py) ----
        topk_key = None
        for key in ['test_topk', 'topk', 'candidates', 'test_candidates']:
            if key in pkl_data:
                topk_key = key
                break
        
        scores_key = None
        for key in ['test_probs', 'test_scores', 'scores', 'probs']:
            if key in pkl_data:
                scores_key = key
                break
        
        if topk_key:
            logger.info(f"Found pre-computed top-K candidates at key: '{topk_key}'")
            test_topk = pkl_data[topk_key]
            # Truncate if longer than n_candidates (same as pkl_to_csv.py)
            test_topk = [list(c[:n_candidates]) for c in test_topk]
            logger.info(f"Loaded candidates for {len(test_topk)} samples")
        elif scores_key:
            logger.info(f"Found scores matrix at key: '{scores_key}', computing top-{n_candidates} indices")
            scores = np.array(pkl_data[scores_key])
            logger.info(f"Scores shape: {scores.shape}")
            
            # Use argpartition for efficient top-K selection (EXACT same as pkl_to_csv.py)
            topk_idx = np.argpartition(-scores, kth=min(n_candidates, scores.shape[1]-1), axis=1)[:, :n_candidates]
            
            # Sort the top-K for stable ranking (EXACT same as pkl_to_csv.py)
            sorted_idx = np.argsort(-scores[np.arange(scores.shape[0])[:, None], topk_idx], axis=1)
            test_topk = topk_idx[np.arange(scores.shape[0])[:, None], sorted_idx]
            
            # Convert to list of lists (NO +1 conversion - indices are already item IDs!)
            test_topk = [indices.tolist() for indices in test_topk]
            logger.info(f"Computed top-{n_candidates} candidates for {len(test_topk)} samples")
        else:
            raise ValueError(f"PKL file must contain either top-K candidates or scores. "
                           f"Found keys: {list(pkl_data.keys())}")
        
        # Verify lengths match
        if len(test_topk) != len(df_test):
            logger.warning(f"PKL has {len(test_topk)} samples but test.csv has {len(df_test)} samples")
            logger.warning(f"Will use minimum: {min(len(test_topk), len(df_test))}")
        
        # ---- Create user_id to PKL index mapping (same as pkl_to_csv.py) ----
        user_key = None
        for key in ['user_ids', 'user_list', 'users', 'test_user_ids']:
            if key in pkl_data:
                user_key = key
                break
        
        if user_key:
            logger.info(f"Found user_id mapping at key: '{user_key}'")
            pkl_user_ids = np.array(pkl_data[user_key])
            # Create mapping from user_id to index in PKL
            user_to_pkl_idx = {uid: idx for idx, uid in enumerate(pkl_user_ids)}
            logger.info(f"Created mapping for {len(user_to_pkl_idx)} users")
        else:
            logger.warning("No user_id mapping found in PKL. Assuming PKL order matches test.csv order.")
            user_to_pkl_idx = None
        
        # ---- Generate CSV rows from test_topk (keep original CSV structure) ----
        logger.info(f"Generating test CSV from top-K candidates...")
        new_rows = []
        
        for idx, row in df_test.iterrows():
            user_id = row['user_id']
            target_item_id = row['item_id']
            
            # Find the correct PKL index for this user
            if user_to_pkl_idx is not None:
                if user_id not in user_to_pkl_idx:
                    logger.warning(f"User {user_id} not found in PKL mapping, skipping")
                    continue
                pkl_idx = user_to_pkl_idx[user_id]
            else:
                # Fallback: assume order matches
                pkl_idx = idx
                if pkl_idx >= len(test_topk):
                    logger.warning(f"PKL index {pkl_idx} out of range, skipping")
                    continue
            
            # Get top-K candidates for this user from PKL (AS-IS, no filtering - same as pkl_to_csv.py)
            candidate_items = list(test_topk[pkl_idx][:n_candidates])
            
            # Create new row with only original test.csv columns
            new_row = {
                'user_id': row['user_id'],
                'item_id': row['item_id'],
                'rating': row['rating'],
                'history_item_id': row['history_item_id'],
                'history_rating': row['history_rating'],
                'candidate_item_id': str(candidate_items)  # Only update this column
            }
            new_rows.append(new_row)
            
            if (len(new_rows)) % 100 == 0:
                logger.info(f"Processed {len(new_rows)}/{len(df_test)} samples")
        
        # Create new dataframe and save (preserving original CSV structure)
        df_new_test = pd.DataFrame(new_rows)
        df_new_test.to_csv(output_path, index=False)
        
        logger.success(f"Generated {len(df_new_test)} test samples from PKL top-K")
        logger.success(f"Output saved to: {output_path}")
        
        # Check how many samples have GT in candidates
        gt_in_candidates_count = 0
        for i in range(len(df_new_test)):
            cands = eval(df_new_test.iloc[i]['candidate_item_id'])
            gt = df_new_test.iloc[i]['item_id']
            if gt in cands:
                gt_in_candidates_count += 1
        
        logger.info(f"Candidates per sample: {n_candidates} (from model's top-K, includes history)")
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


if __name__ == '__main__':
    GenerateEmbeddingTestTask().launch()
