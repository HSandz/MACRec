"""
Dynamic prompt builder for agent tasks.

This module handles on-demand formatting of prompts from minimal CSV data.
Instead of pre-computing all text fields during preprocessing, we format them
just-in-time during task execution.
"""

import ast
import pandas as pd
from typing import Any
from loguru import logger


class PromptBuilder:
    """Build prompts on-demand from minimal CSV data and metadata."""
    
    def __init__(self, data_dir: str, dataset: str = 'ml-100k'):
        """
        Initialize prompt builder with metadata.
        
        Args:
            data_dir: Directory containing item.csv and user.csv
            dataset: Dataset name (for dataset-specific handling)
        """
        self.data_dir = data_dir
        self.dataset = dataset
        
        # Load metadata files (small, kept in memory)
        self.item_df = self._load_item_metadata()
        self.user_df = self._load_user_metadata()
    
    def _load_item_metadata(self) -> pd.DataFrame:
        """Load item metadata from item.csv."""
        import os
        item_path = os.path.join(self.data_dir, 'item.csv')
        
        if not os.path.exists(item_path):
            logger.warning(f"Item metadata not found at {item_path}")
            return None
        
        item_df = pd.read_csv(item_path, index_col=0)
        logger.info(f"Loaded {len(item_df)} items from {item_path}")
        return item_df
    
    def _load_user_metadata(self) -> pd.DataFrame:
        """Load user metadata from user.csv (optional)."""
        import os
        user_path = os.path.join(self.data_dir, 'user.csv')
        
        if not os.path.exists(user_path):
            logger.info(f"No user metadata found at {user_path}, will use placeholder")
            return None
        
        user_df = pd.read_csv(user_path, index_col=0)
        logger.info(f"Loaded {len(user_df)} users from {user_path}")
        return user_df
    
    def _parse_list_field(self, value: Any) -> list:
        """Parse string representation of list back to Python list."""
        if pd.isna(value) or value == 'None' or value == '':
            return []
        
        if isinstance(value, list):
            return value
        
        if isinstance(value, str):
            try:
                # Try to parse as Python literal
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except (ValueError, SyntaxError):
                # If parsing fails, return as-is
                logger.warning(f"Could not parse list field: {value}")
                return []
        
        return []
    
    def get_user_profile(self, user_id: int) -> str:
        """Get formatted user profile string."""
        if self.user_df is not None and user_id in self.user_df.index:
            if 'user_profile' in self.user_df.columns:
                profile = self.user_df.loc[user_id, 'user_profile']
                if pd.notna(profile) and profile != '':
                    return profile
        return "unknown"
    
    def get_item_attributes(self, item_id: int) -> str:
        """Get formatted item attributes string."""
        if self.item_df is not None and item_id in self.item_df.index:
            if 'item_attributes' in self.item_df.columns:
                attrs = self.item_df.loc[item_id, 'item_attributes']
                if pd.notna(attrs) and attrs != '':
                    return attrs
        return f"Item {item_id}: unknown"
    
    def format_history(self, history_ids: Any, history_ratings: Any, history_summary: Any = None, max_his: int = 10) -> str:
        """
        Format historical interactions into prompt text.
        
        Args:
            history_ids: List of historical item IDs (or string representation)
            history_ratings: List of corresponding ratings
            history_summary: List of user comments/summaries (optional, for Amazon dataset)
            max_his: Maximum history length to include
            
        Returns:
            Formatted history string
        """
        # Parse list fields
        history_ids = self._parse_list_field(history_ids)
        history_ratings = self._parse_list_field(history_ratings)
        history_summary = self._parse_list_field(history_summary) if history_summary is not None else None
        
        if not history_ids:
            return "None"
        
        # Take last max_his items
        history_ids = history_ids[-max_his:]
        history_ratings = history_ratings[-max_his:] if len(history_ratings) >= len(history_ids) else history_ratings
        if history_summary is not None:
            history_summary = history_summary[-max_his:] if len(history_summary) >= len(history_ids) else history_summary
        
        # Format each historical interaction
        history_lines = []
        for i, item_id in enumerate(history_ids):
            item_attr = self.get_item_attributes(item_id)
            rating = history_ratings[i] if i < len(history_ratings) else 'unknown'
            
            # Include summary if available (Amazon dataset)
            if history_summary is not None and i < len(history_summary):
                summary = history_summary[i]
                history_lines.append(f"{item_attr}, UserComments: {summary} (rating: {rating})")
            else:
                history_lines.append(f"{item_attr} (rating: {rating})")
        
        return "\n".join(history_lines)
    
    def format_candidates(self, candidate_ids: Any) -> str:
        """
        Format candidate items for ranking tasks.
        
        Args:
            candidate_ids: List of candidate item IDs (or string representation)
            
        Returns:
            Formatted candidate string
        """
        candidate_ids = self._parse_list_field(candidate_ids)
        
        if not candidate_ids:
            return ""
        
        # Format each candidate
        candidate_lines = []
        for item_id in candidate_ids:
            item_attr = self.get_item_attributes(item_id)
            candidate_lines.append(f"{item_id}: {item_attr}")
        
        return "\n".join(candidate_lines)
    
    def build_prompt_fields(self, row: pd.Series, max_his: int = 10) -> dict:
        """
        Build all prompt fields for a data row.
        
        Args:
            row: DataFrame row with user_id, item_id, history_item_id, etc.
            max_his: Maximum history length
            
        Returns:
            Dictionary with formatted fields: user_profile, history, 
            target_item_attributes, candidate_item_attributes
        """
        fields = {}
        
        # User profile
        user_id = row.get('user_id')
        if pd.notna(user_id):
            fields['user_profile'] = self.get_user_profile(user_id)
        else:
            fields['user_profile'] = 'unknown'
        
        # History
        history_ids = row.get('history_item_id')
        history_ratings = row.get('history_rating')
        history_summary = row.get('history_summary')  # Optional, for Amazon dataset
        fields['history'] = self.format_history(history_ids, history_ratings, history_summary, max_his)
        
        # Target item attributes
        item_id = row.get('item_id')
        if pd.notna(item_id):
            fields['target_item_attributes'] = self.get_item_attributes(item_id)
        else:
            fields['target_item_attributes'] = 'unknown'
        
        # Candidate item attributes
        candidate_ids = row.get('candidate_item_id')
        if pd.notna(candidate_ids):
            fields['candidate_item_attributes'] = self.format_candidates(candidate_ids)
        else:
            fields['candidate_item_attributes'] = ''
        
        return fields
