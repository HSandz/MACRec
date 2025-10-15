"""
Utility functions for creating mapping files for EmbeddingRetriever
"""
import os
import json
import pandas as pd
from typing import Optional
from loguru import logger

def create_embedding_mappings(embeddings_dir: str, dataset_name: str = None) -> bool:
    """
    Create user_id and item_id mapping files for EmbeddingRetriever.
    
    Args:
        embeddings_dir: Directory containing embedding CSV files
        dataset_name: Name of the dataset (for logging)
        
    Returns:
        bool: True if mappings were created successfully, False otherwise
    """
    try:
        user_emb_file = os.path.join(embeddings_dir, "user_embeddings.csv")
        item_emb_file = os.path.join(embeddings_dir, "item_embeddings.csv")
        user_map_file = os.path.join(embeddings_dir, "user_id_mapping.json")
        item_map_file = os.path.join(embeddings_dir, "item_id_mapping.json")
        
        # Check if embedding files exist
        if not os.path.isfile(user_emb_file):
            logger.warning(f"User embeddings file not found: {user_emb_file}")
            return False
            
        if not os.path.isfile(item_emb_file):
            logger.warning(f"Item embeddings file not found: {item_emb_file}")
            return False
        
        # Check if mapping files already exist
        if os.path.isfile(user_map_file) and os.path.isfile(item_map_file):
            logger.info(f"Mapping files already exist for {dataset_name or 'dataset'}")
            return True
        
        logger.info(f"Creating mapping files for {dataset_name or 'dataset'}...")
        
        # Create user mapping
        user_df = pd.read_csv(user_emb_file)
        if 'user_id' not in user_df.columns:
            logger.error(f"user_id column not found in {user_emb_file}")
            return False
            
        user_ids = user_df['user_id'].tolist()
        user_mapping = {
            "token": [str(uid) for uid in user_ids],
            "inner_id": list(range(len(user_ids)))
        }
        
        with open(user_map_file, 'w', encoding='utf-8') as f:
            json.dump(user_mapping, f, indent=2)
        
        logger.info(f"Created user mapping: {len(user_ids)} users")
        
        # Create item mapping
        item_df = pd.read_csv(item_emb_file)
        if 'item_id' not in item_df.columns:
            logger.error(f"item_id column not found in {item_emb_file}")
            return False
            
        item_ids = item_df['item_id'].tolist()
        item_mapping = {
            "token": [str(iid) for iid in item_ids],
            "inner_id": list(range(len(item_ids)))
        }
        
        with open(item_map_file, 'w', encoding='utf-8') as f:
            json.dump(item_mapping, f, indent=2)
        
        logger.info(f"Created item mapping: {len(item_ids)} items")
        logger.info(f"Mapping files created successfully for {dataset_name or 'dataset'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create mapping files: {e}")
        return False

def ensure_embedding_mappings(embeddings_dir: str, dataset_name: str = None) -> bool:
    """
    Ensure mapping files exist for EmbeddingRetriever.
    Create them if they don't exist.
    
    Args:
        embeddings_dir: Directory containing embedding CSV files
        dataset_name: Name of the dataset (for logging)
        
    Returns:
        bool: True if mappings exist or were created successfully
    """
    user_map_file = os.path.join(embeddings_dir, "user_id_mapping.json")
    item_map_file = os.path.join(embeddings_dir, "item_id_mapping.json")
    
    # If both mapping files exist, we're good
    if os.path.isfile(user_map_file) and os.path.isfile(item_map_file):
        return True
    
    # Otherwise, try to create them
    return create_embedding_mappings(embeddings_dir, dataset_name)
