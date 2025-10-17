import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import argparse
import sys
import json
from macrec.tools.retriever import Retriever

def load_embeddings(dataset: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict], Optional[Dict]]:
    """Load embeddings and mappings directly from the embeddings folder."""
    embeddings_dir = f"data/{dataset}/embeddings"
    
    if not os.path.exists(embeddings_dir):
        logger.warning(f"Embeddings directory not found: {embeddings_dir}")
        return None, None, None, None
    
    try:
        # Load item embeddings
        item_emb_path = os.path.join(embeddings_dir, "item_embeddings.csv")
        item_emb_df = pd.read_csv(item_emb_path)
        item_embeddings = item_emb_df.iloc[:, 1:].values  # Skip first column (item_id)
        
        # Load user embeddings
        user_emb_path = os.path.join(embeddings_dir, "user_embeddings.csv")
        user_emb_df = pd.read_csv(user_emb_path)
        user_embeddings = user_emb_df.iloc[:, 1:].values  # Skip first column (user_id)
        
        # Load item mapping
        item_map_path = os.path.join(embeddings_dir, "item_id_mapping.json")
        with open(item_map_path, 'r') as f:
            item_mapping = json.load(f)
        
        # Load user mapping
        user_map_path = os.path.join(embeddings_dir, "user_id_mapping.json")
        with open(user_map_path, 'r') as f:
            user_mapping = json.load(f)
        
        logger.info(f"Loaded embeddings: {user_embeddings.shape[0]} users, {item_embeddings.shape[0]} items")
        return user_embeddings, item_embeddings, user_mapping, item_mapping
        
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return None, None, None, None

def load_retriever(dataset: str, top_k: int = 20) -> Retriever:
    config_path = "config/tools/retriever.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Inject dataset-specific overrides
    config['dataset'] = dataset
    config['top_k'] = top_k
    config['base_dir'] = f"data/{dataset}/embeddings"
    config['user_history_path'] = f"data/{dataset}/train.csv"
    config['similarity_file_path'] = f"data/{dataset}/ranked_top_{top_k}.csv"

    return Retriever(config_path=config_path)

def get_users(dataset: str) -> List[int]:
    path = f"data/{dataset}/train.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return sorted(df['user_id'].unique())

def generate_topk_with_embeddings(
    dataset: str, 
    top_k: int = 20, 
    normalize: bool = True,
    exclude_seen: bool = True
) -> pd.DataFrame:
    logger.info(f"Generating {top_k} items for {dataset} using embeddings")
    
    # Load embeddings and mappings
    user_embeddings, item_embeddings, user_mapping, item_mapping = load_embeddings(dataset)
    
    if user_embeddings is None or item_embeddings is None:
        logger.error("Failed to load embeddings")
        return pd.DataFrame()
    
    # Normalize embeddings if requested
    if normalize:
        user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    
    # Load user history for excluding seen items
    user_seen_items = {}
    if exclude_seen:
        train_path = f"data/{dataset}/train.csv"
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            for user_id, group in train_df.groupby('user_id'):
                user_seen_items[user_id] = set(group['item_id'].values)
    
    # Create mapping dictionaries
    user_token_to_inner = {str(token): inner_id for token, inner_id in zip(user_mapping['token'], user_mapping['inner_id'])}
    item_inner_to_token = user_mapping['token']  # This should be item_mapping['token']
    item_inner_to_token = item_mapping['token']
    item_token_to_inner = {str(token): inner_id for token, inner_id in zip(item_mapping['token'], item_mapping['inner_id'])}
    
    # Get all users from training data
    users = get_users(dataset)
    results = []
    
    logger.info(f"Processing {len(users)} users")
    
    for i, user_id in enumerate(users):
        if i % 100 == 0:
            logger.info(f"Processing {i+1}/{len(users)}: {user_id}")
        
        try:
            # Get user inner ID
            user_token = str(user_id)
            if user_token not in user_token_to_inner:
                logger.warning(f"User {user_id} not found in user mapping")
                continue
                
            user_inner = user_token_to_inner[user_token]
            
            # Compute similarity scores
            user_emb = user_embeddings[user_inner:user_inner+1]
            scores = (user_emb @ item_embeddings.T).flatten()
            
            # Exclude seen items if requested
            if exclude_seen and user_id in user_seen_items:
                seen_items = user_seen_items[user_id]
                for item_id in seen_items:
                    item_token = str(item_id)
                    if item_token in item_token_to_inner:
                        item_inner = item_token_to_inner[item_token]
                        scores[item_inner] = -np.inf
            
            # Handle case where all items are excluded
            if np.all(np.isinf(scores)) and np.all(scores < 0):
                logger.warning(f"All items excluded for user {user_id}")
                row = {"user_id": user_id}
                for j in range(top_k):
                    row[f"item_id_{j}"] = None
                results.append(row)
                continue
            
            # Get top-k items
            k = min(top_k, scores.shape[0])
            topk_idx = np.argpartition(-scores, k-1)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
            
            # Convert inner IDs to token IDs
            row = {"user_id": user_id}
            for j in range(top_k):
                if j < len(topk_idx):
                    item_inner = topk_idx[j]
                    item_token = item_inner_to_token[item_inner]
                    try:
                        row[f"item_id_{j}"] = int(item_token)
                    except ValueError:
                        row[f"item_id_{j}"] = item_token
                else:
                    row[f"item_id_{j}"] = None
            results.append(row)
            
        except Exception as e:
            logger.warning(f"Error processing user {user_id}: {e}")
            row = {"user_id": user_id}
            for j in range(top_k):
                row[f"item_id_{j}"] = None
            results.append(row)
    
    df = pd.DataFrame(results)
    logger.success(f"Generated recommendations for {len(results)} users")
    return df

def save_mappings(dataset: str, user_mapping: Dict, item_mapping: Dict) -> None:
    """Save user and item mappings to files."""
    mappings_dir = f"data/{dataset}/mappings"
    os.makedirs(mappings_dir, exist_ok=True)
    
    # Save user mapping
    user_map_path = os.path.join(mappings_dir, "user_mapping.json")
    with open(user_map_path, 'w') as f:
        json.dump(user_mapping, f, indent=2)
    
    # Save item mapping
    item_map_path = os.path.join(mappings_dir, "item_mapping.json")
    with open(item_map_path, 'w') as f:
        json.dump(item_mapping, f, indent=2)
    
    logger.success(f"Saved mappings to {mappings_dir}")

def generate_csv(dataset: str, top_k: int = 20, use_embeddings: bool = True):
    """Generate top@k items CSV file using either embeddings or retriever."""
    logger.info(f"Generating {top_k} items for {dataset}")
    
    if use_embeddings:
        # Use direct embedding approach
        df = generate_topk_with_embeddings(dataset, top_k)
    else:
        # Use retriever approach (original method)
        retriever = load_retriever(dataset, top_k)
        users = get_users(dataset)
        
        logger.info(f"Found {len(users)} users")
        
        results = []
        for i, user_id in enumerate(users):
            if i % 100 == 0:
                logger.info(f"Processing {i+1}/{len(users)}: {user_id}")
            
            try:
                items, _ = retriever.retrieve_embeddings(user_id, k=top_k)
                row = {"user_id": user_id}
                for j in range(top_k):
                    row[f"item_id_{j}"] = items[j] if j < len(items) else None
                results.append(row)
            except:
                row = {"user_id": user_id}
                for j in range(top_k):
                    row[f"item_id_{j}"] = None
                results.append(row)
        
        df = pd.DataFrame(results)
    
    # Save results
    output_file = f"data/{dataset}/ranked_top_{top_k}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.success(f"Generated {output_file}")
    return output_file

def generate_mappings(dataset: str):
    """Generate and save mappings for the dataset."""
    logger.info(f"Generating mappings for {dataset}")
    
    # Load embeddings and mappings
    user_embeddings, item_embeddings, user_mapping, item_mapping = load_embeddings(dataset)
    
    if user_mapping is None or item_mapping is None:
        logger.error("Failed to load mappings")
        return
    
    # Save mappings
    save_mappings(dataset, user_mapping, item_mapping)

def main():
    parser = argparse.ArgumentParser(description="Generate ranked CSV and mappings using embeddings")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--top_k", type=int, default=20, help="Number of items")
    parser.add_argument("--use_embeddings", action="store_true", default=True, help="Use direct embedding approach")
    parser.add_argument("--use_retriever", action="store_true", help="Use retriever approach instead of embeddings")
    parser.add_argument("--generate_mappings", action="store_true", help="Generate and save mappings")
    parser.add_argument("--normalize", action="store_true", default=True, help="Normalize embeddings")
    parser.add_argument("--exclude_seen", action="store_true", default=True, help="Exclude seen items")
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    try:
        # Determine which method to use
        use_embeddings = args.use_embeddings and not args.use_retriever
        
        # Generate top@k items
        output_file = generate_csv(args.dataset, args.top_k, use_embeddings)
        logger.success(f"Completed: {output_file}")
        
        # Generate mappings if requested
        if args.generate_mappings:
            generate_mappings(args.dataset)
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()