import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from loguru import logger

from macrec.tools.base import Tool

class Retriever(Tool):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Use config values directly
        self.base_dir = self.config.get('base_dir')
        self.run_dir = self.config.get('run_dir')
        self.user_emb_file = self.config.get('user_embeddings_file')
        self.item_emb_file = self.config.get('item_embeddings_file')
        self.user_map_file = self.config.get('user_id_map_file')
        self.item_map_file = self.config.get('item_id_map_file')
        self.k = int(self.config.get('top_k', 20))
        self.normalize = bool(self.config.get('normalize', True))
        self.exclude_seen = bool(self.config.get('exclude_seen', True))
        self.history_path = self.config.get('user_history_path')
        self.sim_path = self.config.get('similarity_file_path')
        self.dataset = self.config.get('dataset')
        
        # Load data
        self._emb_loaded = False
        self._sim_loaded = False
        self._user_emb = None
        self._item_emb = None
        self._user_to_inner = None
        self._item_inner_to_token = None
        self._item_token_to_inner = None
        self._user_to_seen = None
        self._sim_data = None
        
        self._load_embeddings()
        if self.sim_path and os.path.exists(self.sim_path):
            self._load_similarity()
    
    def _load_embeddings(self) -> None:
        """Load embedding data."""
        if not os.path.exists(self.base_dir):
            logger.warning(f"Embedding dir not found: {self.base_dir}")
            return
            
        # Find latest run dir
        if self.run_dir is None:
            runs = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
            if not runs:
                logger.warning(f"No runs found in {self.base_dir}")
                return
            self.run_dir = max(runs)
        
        run_path = os.path.join(self.base_dir, self.run_dir)
        
        try:
            # Load embeddings
            self._user_emb = pd.read_csv(os.path.join(run_path, self.user_emb_file), header=None).values
            self._item_emb = pd.read_csv(os.path.join(run_path, self.item_emb_file), header=None).values
            
            # Load mappings
            with open(os.path.join(run_path, self.user_map_file), 'r') as f:
                import json
                user_map = json.load(f)
            with open(os.path.join(run_path, self.item_map_file), 'r') as f:
                item_map = json.load(f)
            
            self._user_to_inner = {str(tok): int(inner) for tok, inner in zip(user_map['token'], user_map['inner_id'])}
            self._item_inner_to_token = [str(tok) for tok in item_map['token']]
            self._item_token_to_inner = {str(tok): idx for idx, tok in enumerate(self._item_inner_to_token)}
            
            if self.normalize:
                self._user_emb = self._user_emb / np.linalg.norm(self._user_emb, axis=1, keepdims=True)
                self._item_emb = self._item_emb / np.linalg.norm(self._item_emb, axis=1, keepdims=True)
            
            self._load_history()
            self._emb_loaded = True
            
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
    
    def _load_history(self) -> None:
        """Load user history for seen items."""
        if not self.history_path or not os.path.exists(self.history_path):
            return
            
        try:
            df = pd.read_csv(self.history_path)
            self._user_to_seen = {}
            for user_id, group in df.groupby('user_id'):
                self._user_to_seen[user_id] = set(group['item_id'].values)
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")
    
    def _load_similarity(self) -> None:
        """Load similarity data."""
        try:
            self._sim_data = pd.read_csv(self.sim_path)
            required_cols = ['user_id'] + [f'item_id_{i}' for i in range(self.k)]
            missing = [c for c in required_cols if c not in self._sim_data.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            self._sim_data = self._sim_data.set_index('user_id')
            self._sim_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load similarity data: {e}")
    
    def reset(self, *args, **kwargs) -> None:
        pass
    
    def retrieve_embeddings(self, user_id: int, k: Optional[int] = None) -> Tuple[List[Union[int, str]], List[float]]:
        """Retrieve top-K items using embeddings."""
        if not self._emb_loaded:
            return [], []
            
        k = k if k is not None else self.k
        if k <= 0:
            return [], []
        
        try:
            user_inner = self._user_to_inner[str(user_id)]
            scores = (self._user_emb[user_inner:user_inner+1] @ self._item_emb.T).flatten()
            
            # Exclude seen items
            if self.exclude_seen and self._user_to_seen and user_id in self._user_to_seen:
                seen_items = self._user_to_seen[user_id]
                for item_id in seen_items:
                    if str(item_id) in self._item_token_to_inner:
                        idx = self._item_token_to_inner[str(item_id)]
                        scores[idx] = -np.inf
            
            if np.all(np.isinf(scores)) and np.all(scores < 0):
                return [], []
            
            k = min(k, scores.shape[0])
            topk_idx = np.argpartition(-scores, k-1)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
            
            topk_tokens = [self._item_inner_to_token[i] for i in topk_idx]
            topk_ids = [int(tok) if tok.isdigit() else tok for tok in topk_tokens]
            topk_scores = scores[topk_idx].tolist()
            
            return topk_ids, topk_scores
            
        except Exception as e:
            logger.warning(f"Failed to retrieve embeddings for user {user_id}: {e}")
            return [], []
    
    def get_similarity_items(self, user_id: int) -> List[Optional[int]]:
        """Get similarity items for user."""
        if not self._sim_loaded:
            return []
            
        try:
            row = self._sim_data.loc[user_id]
            items = []
            for i in range(self.k):
                item = row.get(f'item_id_{i}')
                if item is not None and pd.notna(item):
                    try:
                        items.append(int(item))
                    except:
                        items.append(None)
                else:
                    items.append(None)
            return items
        except:
            return [None] * self.k
    
    def get_for_analysis(self, user_id: int, target_id: Optional[int] = None) -> List[int]:
        """Get items for analysis (prefer similarity, fallback to embeddings)."""
        if self._sim_loaded:
            items = self.get_similarity_items(user_id)
            valid = [i for i in items if i is not None]
        else:
            items, _ = self.retrieve_embeddings(user_id, self.k)
            valid = items if isinstance(items[0], int) else [int(i) for i in items if str(i).isdigit()]
        
        if target_id and target_id not in valid:
            valid.insert(0, target_id)
            if len(valid) > self.k:
                valid = valid[:self.k]
        
        return valid
    
    def get_str_for_analysis(self, user_id: int, target_id: Optional[int] = None) -> str:
        """Get items as formatted string for analysis."""
        items = self.get_for_analysis(user_id, target_id)
        if not items:
            return f"No items for user {user_id}"
        
        formatted = []
        for i, item_id in enumerate(items):
            if i == 0 and target_id and item_id == target_id:
                formatted.append(f"{i+1}. Item {item_id} (Target)")
            else:
                formatted.append(f"{i+1}. Item {item_id}")
        
        return f"Items for user {user_id}:\n" + "\n".join(formatted)
    
    def get_users(self) -> List[int]:
        """Get all user IDs."""
        if self._sim_loaded:
            return self._sim_data.index.tolist()
        elif self._emb_loaded:
            return [int(k) for k in self._user_to_inner.keys()]
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            'emb_loaded': self._emb_loaded,
            'sim_loaded': self._sim_loaded,
            'k': self.k
        }
        
        if self._emb_loaded:
            stats['emb_users'] = len(self._user_to_inner)
            stats['emb_items'] = len(self._item_inner_to_token)
        
        if self._sim_loaded:
            stats['sim_users'] = len(self._sim_data)
            counts = []
            for user_id in self._sim_data.index:
                items = self.get_similarity_items(user_id)
                counts.append(sum(1 for i in items if i is not None))
            if counts:
                stats['avg_items'] = sum(counts) / len(counts)
        
        return stats
