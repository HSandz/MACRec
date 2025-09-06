import os
import json
import numpy as np
from typing import Optional, List, Tuple

from macrec.tools.base import Tool

class EmbeddingRetriever(Tool):
    """
    Retrieve top-K item ids for a given user id from precomputed user/item embeddings
    saved by training (e.g., LightGCN) under run/ directory.

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
        self._token_to_user_inner: Optional[dict[str, int]] = None
        self._item_inner_to_token: Optional[List[str]] = None
        self.top_k: int = int(self.config.get('top_k', 10))
        self.normalize: bool = bool(self.config.get('normalize', True))
        self.exclude_seen: bool = bool(self.config.get('exclude_seen', False))
        self.user_history_path: Optional[str] = self.config.get('user_history_path', None)
        self._user_to_seen_items: Optional[dict[int, set[int]]] = None

    def reset(self, *args, **kwargs) -> None:
        # Stateless for now
        pass

    def _find_latest_run_dir(self, base_dir: str) -> Optional[str]:
        try:
            entries = [
                os.path.join(base_dir, name)
                for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))
            ]
        except Exception:
            return None
        if not entries:
            return None
        entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return entries[0]

    def _load_history(self) -> None:
        if not self.exclude_seen or self.user_history_path is None:
            return
        try:
            import pandas as pd
            df = pd.read_csv(self.user_history_path)
            if 'user_id' in df.columns and 'item_id' in df.columns:
                self._user_to_seen_items = {}
                for uid, iid in zip(df['user_id'].astype(int).tolist(), df['item_id'].astype(int).tolist()):
                    self._user_to_seen_items.setdefault(uid, set()).add(int(iid))
        except Exception:
            # fail open
            self._user_to_seen_items = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        base_dir: str = self.config.get('base_dir', 'lightgcn')
        run_dir: Optional[str] = self.config.get('run_dir')
        if run_dir is None:
            run_dir = self._find_latest_run_dir(base_dir)
            if run_dir is None:
                raise FileNotFoundError(f"No run directory found under {base_dir}")
        else:
            # Join run_dir with base_dir if run_dir is specified
            run_dir = os.path.join(base_dir, run_dir)

        user_emb_file = os.path.join(run_dir, self.config.get('user_embeddings_file', 'user_embeddings.csv'))
        item_emb_file = os.path.join(run_dir, self.config.get('item_embeddings_file', 'item_embeddings.csv'))
        user_map_file = os.path.join(run_dir, self.config.get('user_id_map_file', 'user_id_mapping.json'))
        item_map_file = os.path.join(run_dir, self.config.get('item_id_map_file', 'item_id_mapping.json'))

        # Load user embeddings
        if os.path.isfile(user_emb_file):
            if user_emb_file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(user_emb_file, sep='\t', header=None)
                # The second column contains space-separated embedding values
                embeddings_list = []
                for _, row in df.iterrows():
                    embedding_str = row.iloc[1]  # Second column contains space-separated values
                    embedding_values = list(map(float, embedding_str.split()))
                    embeddings_list.append(embedding_values)
                self._user_embeddings = np.array(embeddings_list, dtype=np.float32)
            else:
                self._user_embeddings = np.load(user_emb_file)
        else:
            raise FileNotFoundError(f"User embeddings not found: {user_emb_file}")

        # Load item embeddings  
        if os.path.isfile(item_emb_file):
            if item_emb_file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(item_emb_file, sep='\t', header=None)
                # The second column contains space-separated embedding values
                embeddings_list = []
                for _, row in df.iterrows():
                    embedding_str = row.iloc[1]  # Second column contains space-separated values
                    embedding_values = list(map(float, embedding_str.split()))
                    embeddings_list.append(embedding_values)
                self._item_embeddings = np.array(embeddings_list, dtype=np.float32)
            else:
                self._item_embeddings = np.load(item_emb_file)
        else:
            raise FileNotFoundError(f"Item embeddings not found: {item_emb_file}")

        with open(user_map_file, 'r', encoding='utf-8') as f:
            user_map = json.load(f)
        with open(item_map_file, 'r', encoding='utf-8') as f:
            item_map = json.load(f)

        # Build mappings
        user_tokens: List[str] = list(map(str, user_map['token']))
        user_inners: List[int] = list(map(int, user_map['inner_id']))
        self._token_to_user_inner = {tok: inner for tok, inner in zip(user_tokens, user_inners)}

        item_tokens: List[str] = list(map(str, item_map['token']))
        # Keep in array index-able by inner id
        self._item_inner_to_token = item_tokens

        if self.normalize:
            def _l2norm(x: np.ndarray) -> np.ndarray:
                denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
                return x / denom
            self._user_embeddings = _l2norm(self._user_embeddings)
            self._item_embeddings = _l2norm(self._item_embeddings)

        self._load_history()
        self._loaded = True

    def _lookup_user_inner(self, user_id: int) -> int:
        assert self._token_to_user_inner is not None
        inner = self._token_to_user_inner.get(str(user_id))
        if inner is None:
            # Some datasets may have tokens as ints without padding
            inner = self._token_to_user_inner.get(user_id)  # type: ignore[index]
        if inner is None:
            raise KeyError(f"User id {user_id} not found in mapping")
        return int(inner)

    def _score_items_for_user(self, user_inner: int) -> np.ndarray:
        assert self._user_embeddings is not None and self._item_embeddings is not None
        u = self._user_embeddings[user_inner: user_inner + 1]  # (1, d)
        # dot product similarity
        return (u @ self._item_embeddings.T).reshape(-1)

    def retrieve(self, user_id: int, k: Optional[int] = None) -> Tuple[List[int], List[float]]:
        """Return top-K item ids (original tokens as ints when possible) and scores."""
        self._ensure_loaded()
        if k is None:
            k = self.top_k
        user_inner = self._lookup_user_inner(int(user_id))
        scores = self._score_items_for_user(user_inner)

        # Exclude seen items if configured
        if self.exclude_seen and self._user_to_seen_items is not None:
            seen = self._user_to_seen_items.get(int(user_id), set())
        else:
            seen = set()

        if seen:
            # set scores of seen items to -inf
            # Map seen tokens back to inner ids
            seen_inners = set()
            if self._item_inner_to_token is not None:
                tok_to_inner = {tok: idx for idx, tok in enumerate(self._item_inner_to_token)}
                for tok in map(str, seen):
                    idx = tok_to_inner.get(tok)
                    if idx is not None:
                        seen_inners.add(idx)
            for idx in seen_inners:
                scores[idx] = -1e30

        topk_idx = np.argpartition(-scores, kth=min(k, scores.shape[0]-1))[:k]
        # sort topk
        topk_idx = topk_idx[np.argsort(-scores[topk_idx])]

        # Map inners to tokens
        assert self._item_inner_to_token is not None
        topk_tokens = [self._item_inner_to_token[i] for i in topk_idx.tolist()]
        # Cast to ints when possible
        def _maybe_int(x: str) -> int:
            try:
                return int(x)
            except Exception:
                return int(x) if x.isdigit() else x  # type: ignore[return-value]
        topk_ids = []
        for tok in topk_tokens:
            try:
                topk_ids.append(int(tok))
            except Exception:
                # keep string token if cannot cast
                # type: ignore[arg-type]
                topk_ids.append(tok)  # type: ignore[list-item]
        topk_scores = scores[topk_idx].tolist()
        return topk_ids, topk_scores

    def retrieve_str(self, user_id: int, k: Optional[int] = None) -> str:
        ids, _ = self.retrieve(user_id=user_id, k=k)
        return f"Retrieved {len(ids)} items for user {user_id}: {', '.join(map(str, ids))}"