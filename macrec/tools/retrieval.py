import pandas as pd
import numpy as np
import os
from loguru import logger
import argparse
from macrec.tools.base import Tool

class RetrievalTool(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_dir = self.config.get('base_dir', 'data/{dataset}/embeddings')
        self.user_embeddings_file = self.config.get('user_embeddings_file', 'user_embeddings.csv')
        self.item_embeddings_file = self.config.get('item_embeddings_file', 'item_embeddings.csv')
        self.user_history_path = self.config.get('user_history_path', 'data/{dataset}/test.csv')
        self.top_k = self.config.get('top_k', 20)
        self.normalize = self.config.get('normalize', True)
        self.exclude_seen = self.config.get('exclude_seen', True)
        
        self.dataset = None
        self.ranked_items = None

    def reset(self, *args, **kwargs) -> None:
        self.dataset = None
        self.ranked_items = None

    def set_dataset(self, dataset: str) -> None:
        self.dataset = dataset
        self.base_dir = self.base_dir.replace('{dataset}', dataset)
        self.user_history_path = self.user_history_path.replace('{dataset}', dataset)

    def generate_ranked_csv(self, dataset: str) -> str:
        """Generate ranked_20.csv with top 20 items per user."""
        try:
            self.set_dataset(dataset)
            
            # Load embeddings
            user_emb_path = os.path.join(self.base_dir, self.user_embeddings_file)
            item_emb_path = os.path.join(self.base_dir, self.item_embeddings_file)
            
            user_df = pd.read_csv(user_emb_path)
            item_df = pd.read_csv(item_emb_path)
            
            # Extract embeddings
            user_ids = user_df['user_id'].values
            item_ids = item_df['item_id'].values
            
            user_embeddings = user_df.drop('user_id', axis=1).values
            item_embeddings = item_df.drop('item_id', axis=1).values
            
            # Normalize if required
            if self.normalize:
                user_embeddings = user_embeddings / (np.linalg.norm(user_embeddings, axis=1, keepdims=True) + 1e-9)
                item_embeddings = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-9)
            
            # Load user history if exclude_seen is True
            seen_items = {}
            if self.exclude_seen and os.path.exists(self.user_history_path):
                history_df = pd.read_csv(self.user_history_path)
                if 'user_id' in history_df.columns and 'item_id' in history_df.columns:
                    for uid in user_ids:
                        user_hist = history_df[history_df['user_id'] == uid]['item_id'].values
                        seen_items[uid] = set(user_hist)
            
            # Compute cosine similarity and get top-k items for each user
            results = []
            for i, uid in enumerate(user_ids):
                scores = np.dot(item_embeddings, user_embeddings[i])
                
                # Exclude seen items
                if uid in seen_items:
                    for j, iid in enumerate(item_ids):
                        if iid in seen_items[uid]:
                            scores[j] = -np.inf
                
                # Get top-k items
                top_indices = np.argsort(scores)[::-1][:self.top_k]
                top_items = item_ids[top_indices]
                
                results.append([uid] + list(top_items))
            
            # Create dataframe
            columns = ['user_id'] + [f'top_{i+1}' for i in range(self.top_k)]
            ranked_df = pd.DataFrame(results, columns=columns)
            
            # Save to file
            output_path = f"data/{dataset}/ranked_20.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ranked_df.to_csv(output_path, index=False)
            
            logger.info(f"Generated ranked_20.csv with {len(ranked_df)} users")
            return f"Successfully generated {output_path} with {len(ranked_df)} users"
        
        except Exception as e:
            logger.error(f"Error in generate_ranked_csv: {e}")
            return f"Error generating ranked CSV: {str(e)}"

    def get_top_items(self, user_id: int, dataset: str = None) -> list:
        """Get top 20 items for a specific user from ranked_20.csv."""
        try:
            if dataset:
                self.set_dataset(dataset)
            
            if self.ranked_items is None:
                ranked_path = f"data/{self.dataset}/ranked_20.csv"
                if not os.path.exists(ranked_path):
                    return []
                self.ranked_items = pd.read_csv(ranked_path)
            
            user_row = self.ranked_items[self.ranked_items['user_id'] == user_id]
            if user_row.empty:
                return []
            
            # Get top 20 items (columns top_1 to top_20)
            top_items = user_row.iloc[0, 1:21].values.tolist()
            return [int(item) for item in top_items if pd.notna(item)]
        
        except Exception as e:
            logger.error(f"Error in get_top_items: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description='Generate ranked_20.csv from embeddings')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ml-100k)')
    parser.add_argument('--config', type=str, default='config/tools/retrieval.json', help='Retrieval tool config')
    args = parser.parse_args()
    tool = RetrievalTool(config_path=args.config)
    print(f"Generating ranked_20.csv for dataset: {args.dataset}")
    result = tool.generate_ranked_csv(args.dataset)
    print(result)

if __name__ == '__main__':
    main()