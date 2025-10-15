import os
import json
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union, Dict

from macrec.tools.base import Tool

class EmbeddingRetriever(Tool):
    """
    Retrieve top-K item ids for a given user id from precomputed user/item embeddings
    saved by training (e.g., LightGCN) under lightgcn/output/ directory.

    Config schema (JSON at config_path):
    - base_dir: str, directory containing run outputs (e.g., "lightgcn")
    - run_dir: Optional[str], specific run subdirectory; if not provided, will pick the latest
    - user_embeddings_file: str, default "user_embeddings.csv"
    - item_embeddings_file: str, default "item_embeddings.csv"
    - user_id_map_file: str, default "user_id_mapping.json"
    - item_id_map_file: str, default "item_id_mapping.json"
    - top_k: int, default 10
    - normalize: bool, default True (l2 normalize embeddings before dot-product)
    - exclude_seen: bool, default False (requires providing user_history_path)
    - user_history_path: Optional[str], CSV with at least columns [user_id, item_id] to exclude
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._loaded = False
        self._user_embeddings: Optional[np.ndarray] = None
        self._item_embeddings: Optional[np.ndarray] = None
        self._token_to_user_inner: Optional[Dict[str, int]] = None
        self._item_inner_to_token: Optional[List[str]] = None
        self._item_token_to_inner: Optional[Dict[str, int]] = None
        self.top_k: int = int(self.config.get('top_k', 10))
        self.normalize: bool = bool(self.config.get('normalize', True))
        self.exclude_seen: bool = bool(self.config.get('exclude_seen', True))
        self.user_history_path: Optional[str] = self.config.get('user_history_path', None)
        self._user_to_seen_items: Optional[Dict[int, set[int]]] = None

    def reset(self) -> None:
        """Reset the retriever to unloaded state, clearing all cached data."""
        self._loaded = False
        self._user_embeddings = None
        self._item_embeddings = None
        self._token_to_user_inner = None
        self._item_inner_to_token = None
        self._item_token_to_inner = None
        self._user_to_seen_items = None

    def _find_latest_run_dir(self, base_dir: str) -> Optional[str]:
        if not os.path.isdir(base_dir):
            return None
        try:
            entries = [
                os.path.join(base_dir, name)
                for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))
            ]
        except OSError:
            return None
        if not entries:
            return None
        entries.sort(key=os.path.getmtime, reverse=True)
        return entries[0]

    def _load_history(self) -> None:
        if not self.exclude_seen or not self.user_history_path or not os.path.isfile(self.user_history_path):
            return
        try:
            df = pd.read_csv(self.user_history_path)
            if 'user_id' in df.columns and 'item_id' in df.columns:
                self._user_to_seen_items = (
                    df.groupby('user_id')['item_id']
                    .apply(set)
                    .to_dict()
                )
        except Exception:
            # fail open
            self._user_to_seen_items = None

    def _load_embeddings_from_csv(self, file_path: str) -> np.ndarray:
        try:
            # Try to read with comma separator and header first (new format)
            try:
                df = pd.read_csv(file_path, sep=',', header=0)
                if df.shape[1] < 2:
                    raise ValueError(f"CSV file {file_path} must have at least 2 columns")
                
                # Skip the first column (ID column) and get embedding values
                embedding_columns = df.columns[1:]  # All columns except the first (ID)
                embeddings = df[embedding_columns].values.astype(np.float32)
                return embeddings
                
            except Exception:
                # Fallback to original tab-separated format without header
                df = pd.read_csv(file_path, sep='\t', header=None)
                if df.shape[1] < 2:
                    raise ValueError(f"CSV file {file_path} must have at least 2 columns")
                
                # The second column contains space-separated embedding values
                def _parse_embedding_values(x):
                    try:
                        return [float(v) for v in x]
                    except ValueError as e:
                        raise ValueError(f"Invalid embedding value in {file_path}: {e}")
                
                embeddings_series = df.iloc[:, 1].str.split().apply(_parse_embedding_values)
                return np.array(embeddings_series.tolist(), dtype=np.float32)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings from {file_path}: {e}")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        base_dir: str = self.config.get('base_dir', 'lightgcn')
        run_dir_config: Optional[str] = self.config.get('run_dir')
        
        if run_dir_config:
            run_dir = os.path.join(base_dir, run_dir_config)
        else:
            run_dir = self._find_latest_run_dir(base_dir)

        if not run_dir or not os.path.isdir(run_dir):
            raise FileNotFoundError(f"No valid run directory found under {base_dir}")

        user_emb_file = os.path.join(run_dir, self.config.get('user_embeddings_file', 'user_embeddings.csv'))
        item_emb_file = os.path.join(run_dir, self.config.get('item_embeddings_file', 'item_embeddings.csv'))
        user_map_file = os.path.join(run_dir, self.config.get('user_id_map_file', 'user_id_mapping.json'))
        item_map_file = os.path.join(run_dir, self.config.get('item_id_map_file', 'item_id_mapping.json'))

        # Load user embeddings
        if os.path.isfile(user_emb_file):
            if user_emb_file.endswith('.csv'):
                self._user_embeddings = self._load_embeddings_from_csv(user_emb_file)
            else:
                self._user_embeddings = np.load(user_emb_file)
        else:
            raise FileNotFoundError(f"User embeddings not found: {user_emb_file}")

        # Load item embeddings  
        if os.path.isfile(item_emb_file):
            if item_emb_file.endswith('.csv'):
                self._item_embeddings = self._load_embeddings_from_csv(item_emb_file)
            else:
                self._item_embeddings = np.load(item_emb_file)
        else:
            raise FileNotFoundError(f"Item embeddings not found: {item_emb_file}")

        with open(user_map_file, 'r', encoding='utf-8') as f:
            user_map = json.load(f)
        with open(item_map_file, 'r', encoding='utf-8') as f:
            item_map = json.load(f)

        # Build mappings
        self._token_to_user_inner = {str(tok): int(inner) for tok, inner in zip(user_map['token'], user_map['inner_id'])}
        self._item_inner_to_token = [str(tok) for tok in item_map['token']]
        self._item_token_to_inner = {str(tok): idx for idx, tok in enumerate(self._item_inner_to_token)}

        if self.normalize:
            def _l2norm(x: np.ndarray) -> np.ndarray:
                denom = np.linalg.norm(x, axis=1, keepdims=True)
                return np.divide(x, denom, out=np.zeros_like(x), where=denom!=0)
            self._user_embeddings = _l2norm(self._user_embeddings)
            self._item_embeddings = _l2norm(self._item_embeddings)

        self._load_history()
        self._loaded = True

    def _lookup_user_inner(self, user_id: int) -> int:
        if self._token_to_user_inner is None:
            raise RuntimeError("User mappings not loaded. Call _ensure_loaded() first.")
        inner = self._token_to_user_inner.get(str(user_id))
        if inner is None:
            raise KeyError(f"User id {user_id} not found in mapping. Available users: {list(self._token_to_user_inner.keys())[:10]}...")
        return inner

    def _score_items_for_user(self, user_inner: int) -> np.ndarray:
        if self._user_embeddings is None or self._item_embeddings is None:
            raise RuntimeError("Embeddings not loaded. Call _ensure_loaded() first.")
        if user_inner >= self._user_embeddings.shape[0]:
            raise IndexError(f"User inner id {user_inner} out of range. Max: {self._user_embeddings.shape[0] - 1}")
        
        u = self._user_embeddings[user_inner: user_inner + 1]  # (1, d)
        # dot product similarity
        return (u @ self._item_embeddings.T).flatten()

    def retrieve(self, user_id: int, k: Optional[int] = None) -> Tuple[List[Union[int, str]], List[float]]:
        """Return top-K item ids (original tokens as ints when possible) and scores."""
        self._ensure_loaded()
        k = k if k is not None else self.top_k
        
        # Validate input parameters
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        user_inner = self._lookup_user_inner(user_id)
        scores = self._score_items_for_user(user_inner)

        # Exclude seen items if configured
        if self.exclude_seen and self._user_to_seen_items and self._item_token_to_inner:
            seen_items = self._user_to_seen_items.get(user_id, set())
            if seen_items:
                seen_inners = {self._item_token_to_inner.get(str(item_id)) for item_id in seen_items}
                for idx in seen_inners:
                    if idx is not None:
                        scores[idx] = -np.inf

        # Check if all scores are -inf (all items excluded)
        if np.all(np.isinf(scores)) and np.all(scores < 0):
            return [], []

        # Ensure k is not larger than the number of items
        k = min(k, scores.shape[0])
        
        # Get top-k indices
        topk_idx = np.argpartition(-scores, k - 1)[:k]
        topk_idx = topk_idx[np.argsort(-scores[topk_idx])]

        # Map inners to tokens
        if self._item_inner_to_token is None:
            raise RuntimeError("Item mappings not loaded. Call _ensure_loaded() first.")
        topk_tokens = [self._item_inner_to_token[i] for i in topk_idx]
        
        def _maybe_int(token: str) -> Union[int, str]:
            return int(token) if token.isdigit() else token

        topk_ids = [_maybe_int(tok) for tok in topk_tokens]
        topk_scores = scores[topk_idx].tolist()
        
        return topk_ids, topk_scores

    def retrieve_str(self, user_id: int, k: Optional[int] = None) -> str:
        ids, _ = self.retrieve(user_id=user_id, k=k)
        return f"Retrieved {len(ids)} items for user {user_id}: {', '.join(map(str, ids))}"