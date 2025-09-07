#!/usr/bin/env python3
"""
FIXED VERSION: Simple script to evaluate HitRatio and NDCG [1,3,5] using computed embeddings
from lightgcn/output/ and ground truth from processed test data

MAJOR FIXES:
1. Fixed ID mapping logic (inner_id vs original_id)
2. Proper embedding loading with reverse mapping
3. Corrected test data loading and history parsing
4. Improved performance with batching
5. Better error handling and validation
"""

import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from macrec.evaluation.rank_metric import HitRatioAt, NDCGAt
import argparse
import os
from loguru import logger

def load_embeddings_fixed(embedding_file, mapping_file):
    """Load embeddings with CORRECTED ID mapping logic"""
    logger.info(f"Loading embeddings from {embedding_file}...")
    
    # Load mapping first to understand the structure
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Create reverse mapping: inner_id -> original_id
    inner_to_original = {idx: original_id for idx, original_id in enumerate(mapping['inner_id'])}
    
    # Load embeddings (format: inner_id \t embedding_values...)
    embeddings_data = []
    inner_ids = []
    
    with open(embedding_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                inner_id = int(parts[0])
                embedding = [float(x) for x in parts[1:]]
                embeddings_data.append(embedding)
                inner_ids.append(inner_id)
    
    # Convert to numpy array
    embeddings = np.array(embeddings_data)
    
    # Create mapping from original_id to embedding index
    original_id_to_idx = {}
    for idx, inner_id in enumerate(inner_ids):
        if inner_id in inner_to_original:
            original_id = inner_to_original[inner_id]
            original_id_to_idx[original_id] = idx
    
    logger.info(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    logger.info(f"Created mapping for {len(original_id_to_idx)} original IDs")
    return embeddings, original_id_to_idx

def load_test_data_fixed(test_file):
    """Load test data with proper validation"""
    if not os.path.exists(test_file):
        logger.error(f"Test file {test_file} not found!")
        logger.info("Please run preprocessing first: python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k")
        raise FileNotFoundError(f"Test file {test_file} not found")
    
    logger.info(f"Loading test data from {test_file}...")
    df = pd.read_csv(test_file)
    
    # Validate required columns
    required_cols = ['user_id', 'item_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in test data: {missing_cols}")
    
    # Extract user-item pairs for evaluation
    test_interactions = []
    for _, row in df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        test_interactions.append((user_id, item_id))
    
    logger.info(f"Loaded {len(test_interactions)} test interactions")
    logger.info(f"Unique users: {df['user_id'].nunique()}, Unique items: {df['item_id'].nunique()}")
    return test_interactions, df

def get_user_history_fixed(test_df, user_id):
    """Get user's history with improved parsing"""
    user_data = test_df[test_df['user_id'] == user_id]
    
    history = []
    if not user_data.empty:
        # Try different possible column names for history
        history_cols = ['history_item_id', 'history', 'user_history']
        history_col = None
        
        for col in history_cols:
            if col in user_data.columns:
                history_col = col
                break
        
        if history_col and pd.notna(user_data.iloc[0][history_col]):
            history_str = str(user_data.iloc[0][history_col])
            
            # Handle different formats: "[1,2,3]", "1,2,3", or actual list
            if isinstance(history_str, str):
                if history_str.startswith('[') and history_str.endswith(']'):
                    history_str = history_str.strip('[]')
                
                if history_str and history_str != 'None':
                    try:
                        history = [int(x.strip()) for x in history_str.split(',') if x.strip()]
                    except ValueError as e:
                        logger.warning(f"Could not parse history for user {user_id}: {history_str}")
    
    return history

def compute_rankings_batched(user_embeddings, item_embeddings, user_id_to_idx, item_id_to_idx, 
                           test_interactions, test_df, top_k=100, batch_size=100):
    """Compute rankings with batching for better performance"""
    logger.info("Computing rankings with batching...")
    
    rankings = {}
    processed_users = set()
    
    # Get all unique users
    unique_users = list(set([user_id for user_id, _ in test_interactions]))
    logger.info(f"Processing {len(unique_users)} unique users...")
    
    # Process users in batches
    for i in range(0, len(unique_users), batch_size):
        batch_users = unique_users[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(unique_users) + batch_size - 1)//batch_size}")
        
        for user_id in batch_users:
            if user_id in processed_users or user_id not in user_id_to_idx:
                continue
            
            # Get user embedding
            user_idx = user_id_to_idx[user_id]
            user_emb = user_embeddings[user_idx].reshape(1, -1)
            
            # Get user's history to exclude from recommendations
            history = get_user_history_fixed(test_df, user_id)
            history_set = set(history)
            
            # Get valid items (exclude history items)
            valid_items = [(item_id, item_idx) for item_id, item_idx in item_id_to_idx.items() 
                          if item_id not in history_set]
            
            if not valid_items:
                logger.warning(f"No valid items for user {user_id}")
                continue
            
            # Extract item embeddings for valid items
            valid_item_indices = [item_idx for _, item_idx in valid_items]
            valid_item_ids = [item_id for item_id, _ in valid_items]
            valid_item_embeddings = item_embeddings[valid_item_indices]
            
            # Compute similarities
            similarities = cosine_similarity(user_emb, valid_item_embeddings)[0]
            
            # Sort by similarity and get top-k
            item_sim_pairs = list(zip(valid_item_ids, similarities))
            item_sim_pairs.sort(key=lambda x: x[1], reverse=True)
            top_items = [item_id for item_id, _ in item_sim_pairs[:top_k]]
            
            rankings[user_id] = top_items
            processed_users.add(user_id)
    
    logger.info(f"Computed rankings for {len(rankings)} users")
    return rankings

def evaluate_metrics_fixed(rankings, test_interactions, topks=[1, 3, 5]):
    """Evaluate metrics with better error handling"""
    logger.info("Evaluating metrics...")
    
    # Initialize metrics
    hr_metrics = HitRatioAt(topks)
    ndcg_metrics = NDCGAt(topks)
    
    total_interactions = len(test_interactions)
    valid_interactions = 0
    
    # Evaluate each test interaction
    for user_id, target_item in test_interactions:
        if user_id in rankings:
            predicted_ranking = rankings[user_id]
            
            # Update metrics
            hr_metrics.update({'answer': predicted_ranking, 'label': target_item})
            ndcg_metrics.update({'answer': predicted_ranking, 'label': target_item})
            valid_interactions += 1
    
    logger.info(f"Evaluated {valid_interactions}/{total_interactions} interactions")
    
    # Compute final metrics
    hr_results = hr_metrics.compute()
    ndcg_results = ndcg_metrics.compute()
    
    return hr_results, ndcg_results

def main():
    parser = argparse.ArgumentParser(description='FIXED: Evaluate embeddings using HitRatio and NDCG')
    parser.add_argument('--user_embeddings', default='lightgcn/output/user_embeddings.csv',
                       help='Path to user embeddings file')
    parser.add_argument('--item_embeddings', default='lightgcn/output/item_embeddings.csv',
                       help='Path to item embeddings file')
    parser.add_argument('--user_mapping', default='lightgcn/output/user_id_mapping.json',
                       help='Path to user ID mapping file')
    parser.add_argument('--item_mapping', default='lightgcn/output/item_id_mapping.json',
                       help='Path to item ID mapping file')
    parser.add_argument('--test_file', default='data/ml-100k/test.csv',
                       help='Path to test data file')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of top items to consider for ranking')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for processing users')
    
    args = parser.parse_args()
    
    # Check if files exist
    required_files = [args.user_embeddings, args.item_embeddings, 
                     args.user_mapping, args.item_mapping]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Error: File {file_path} not found!")
            return
    
    try:
        # Load embeddings with FIXED logic
        logger.info("Loading embeddings with corrected ID mapping...")
        user_embeddings, user_id_to_idx = load_embeddings_fixed(args.user_embeddings, args.user_mapping)
        item_embeddings, item_id_to_idx = load_embeddings_fixed(args.item_embeddings, args.item_mapping)
        
        # Load test data with validation
        test_interactions, test_df = load_test_data_fixed(args.test_file)
        
        # Compute rankings with batching
        rankings = compute_rankings_batched(user_embeddings, item_embeddings, 
                                          user_id_to_idx, item_id_to_idx, 
                                          test_interactions, test_df, 
                                          args.top_k, args.batch_size)
        
        # Evaluate metrics
        hr_results, ndcg_results = evaluate_metrics_fixed(rankings, test_interactions)
        
        # Print results
        print("\n" + "="*60)
        print("FIXED EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nDataset Statistics:")
        print(f"  Total test interactions: {len(test_interactions)}")
        print(f"  Users with rankings: {len(rankings)}")
        print(f"  Coverage: {len(rankings)/len(set([u for u, _ in test_interactions]))*100:.1f}%")
        
        print("\nHitRatio Results:")
        for metric, value in hr_results.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nNDCG Results:")
        for metric, value in ndcg_results.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.info("\nTroubleshooting steps:")
        logger.info("1. Make sure you have run preprocessing: python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k")
        logger.info("2. Check that LightGCN training has generated embeddings in lightgcn/output/")
        logger.info("3. Verify file paths and permissions")
        raise

if __name__ == "__main__":
    main()