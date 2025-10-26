import os
import json
import random
import pandas as pd
import numpy as np
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.utils import append_his_info

def read_data(dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read Yelp2018 raw data files."""
    logger.info('Reading Yelp2018 review file...')
    review_file = os.path.join(dir, 'yelp_academic_dataset_review.json')
    
    reviews = []
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            reviews.append({
                'user_id': review['user_id'],
                'business_id': review['business_id'],
                'rating': review['stars'],
                'timestamp': review['date']
            })
    
    data_df = pd.DataFrame(reviews)
    logger.info(f'Successfully read {data_df.shape[0]} reviews')
    
    logger.info('Reading Yelp2018 business file...')
    business_file = os.path.join(dir, 'yelp_academic_dataset_business.json')
    businesses = []
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            business = json.loads(line)
            businesses.append({
                'business_id': business['business_id'],
                'name': business.get('name', 'Unknown'),
                'categories': business.get('categories', 'Unknown'),
                'city': business.get('city', 'Unknown'),
                'state': business.get('state', 'Unknown')
            })
    
    business_df = pd.DataFrame(businesses)
    logger.info(f'Successfully read {business_df.shape[0]} businesses')
    
    logger.info('Reading Yelp2018 user file...')
    user_file = os.path.join(dir, 'yelp_academic_dataset_user.json')
    users = []
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            user = json.loads(line)
            users.append({
                'user_id': user['user_id'],
                'review_count': user.get('review_count', 0),
                'average_stars': user.get('average_stars', 0.0)
            })
    
    user_df = pd.DataFrame(users)
    logger.info(f'Successfully read {user_df.shape[0]} users')
    
    return data_df, business_df, user_df

def process_user_data(user_df: pd.DataFrame, filtered_data_df: pd.DataFrame) -> pd.DataFrame:
    """Process user data to create user profiles."""
    # Filter users that are in the filtered dataset
    valid_users = filtered_data_df['user_id'].unique()
    user_df = user_df[user_df['user_id'].isin(valid_users)].copy()
    user_df = user_df.set_index('user_id')
    
    template = PromptTemplate(
        template='User with {review_count} reviews, average rating: {average_stars:.1f}',
        input_variables=['review_count', 'average_stars'],
    )
    user_df['user_profile'] = user_df.apply(lambda x: template.format(**x), axis=1)
    
    return user_df

def process_item_data(business_df: pd.DataFrame, filtered_data_df: pd.DataFrame) -> pd.DataFrame:
    """Process business data to create item attributes."""
    # Filter businesses that are in the filtered dataset
    valid_businesses = filtered_data_df['item_id'].unique()
    business_df = business_df[business_df['business_id'].isin(valid_businesses)].copy()
    business_df = business_df.rename(columns={'business_id': 'item_id'})
    business_df = business_df.set_index('item_id')
    
    template = PromptTemplate(
        template='Business: {name}, Categories: {categories}, Location: {city}, {state}',
        input_variables=['name', 'categories', 'city', 'state'],
    )
    business_df['item_attributes'] = business_df.apply(lambda x: template.format(**x), axis=1)
    
    return business_df

def filter_data(data_df: pd.DataFrame, min_interactions: int = 10) -> pd.DataFrame:
    """Apply k-core filtering."""
    original_size = data_df.shape[0]
    original_users = data_df['user_id'].nunique()
    original_items = data_df['business_id'].nunique()
    
    filter_before = -1
    while filter_before != data_df.shape[0]:
        filter_before = data_df.shape[0]
        data_df = data_df.groupby('user_id').filter(lambda x: len(x) >= min_interactions)
        data_df = data_df.groupby('business_id').filter(lambda x: len(x) >= min_interactions)
    
    filtered_size = data_df.shape[0]
    filtered_users = data_df['user_id'].nunique()
    filtered_items = data_df['business_id'].nunique()
    
    logger.info(f'{min_interactions}-core filtering: {original_size} -> {filtered_size} interactions')
    logger.info(f'Users: {original_users} -> {filtered_users}, Businesses: {original_items} -> {filtered_items}')
    
    return data_df

def process_interaction_data(data_df: pd.DataFrame, n_neg_items: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process interaction data with negative sampling."""
    # Convert timestamp
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df['timestamp'] = data_df['timestamp'].astype(np.int64) // 10**9
    
    # Sort by timestamp
    data_df = data_df.sort_values(by=['timestamp'])
    data_df = filter_data(data_df)
    
    # Rename business_id to item_id
    data_df = data_df.rename(columns={'business_id': 'item_id'})
    
    # Build clicked item set
    clicked_item_set = {}
    for uid, group in data_df.groupby('user_id'):
        clicked_item_set[uid] = set(group['item_id'].values)
    
    # Get all valid items
    all_items = data_df['item_id'].unique()
    
    def negative_sample(df):
        neg_items = []
        for uid in df['user_id'].values:
            user_clicked = clicked_item_set[uid]
            user_neg_items = []
            while len(user_neg_items) < n_neg_items:
                neg_item = np.random.choice(all_items)
                if neg_item not in user_clicked and neg_item not in user_neg_items:
                    user_neg_items.append(neg_item)
            neg_items.append(user_neg_items)
        df['neg_item_id'] = neg_items
        return df
    
    def generate_dev_test(data_df: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df
    
    # Split data
    leave_df = data_df.groupby('user_id').head(1)
    left_df = data_df.drop(leave_df.index)
    
    [test_df, dev_df], train_df = generate_dev_test(left_df)
    train_df = pd.concat([leave_df, train_df]).sort_index()
    
    # Apply negative sampling
    train_df = negative_sample(train_df)
    dev_df = negative_sample(dev_df)
    test_df = negative_sample(test_df)
    
    logger.info(f'Data split - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}')
    
    return train_df, dev_df, test_df, data_df

def process_data(dir: str, n_neg_items: int = 7, k_core: int = 10):
    """Process Yelp2018 dataset."""
    logger.info(f'Starting to process Yelp2018 dataset in {dir}')
    
    raw_data_dir = os.path.join(dir, "raw")
    
    data_df, business_df, user_df = read_data(raw_data_dir)
    
    logger.info('Processing interaction data...')
    train_df, dev_df, test_df, filtered_data_df = process_interaction_data(data_df, n_neg_items)
    
    logger.info('Processing user data...')
    user_df = process_user_data(user_df, filtered_data_df)
    logger.info(f'Number of users: {user_df.shape[0]}')
    
    logger.info('Processing business data...')
    item_df = process_item_data(business_df, filtered_data_df)
    logger.info(f'Number of businesses: {item_df.shape[0]}')
    
    logger.info('Appending history information...')
    dfs = append_his_info([train_df, dev_df, test_df], neg=True)
    
    for df in dfs:
        df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
        df['history'] = df.apply(lambda x: [f'{item_attributes} (rating: {rating})' for item_attributes, rating in zip(x['history'], x['history_rating'])], axis=1)
        df['history'] = df['history'].apply(lambda x: '\n'.join(x))
        df['user_profile'] = df['user_id'].apply(lambda x: user_df.loc[x]['user_profile'])
        df['target_item_attributes'] = df['item_id'].apply(lambda x: item_df.loc[x]['item_attributes'])
        df['candidate_item_id'] = df.apply(lambda x: [x['item_id']]+x['neg_item_id'], axis=1)
        df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: random.sample(x, len(x)))
        
        def get_candidate_attributes(candidate_ids):
            result = []
            for item_id in candidate_ids:
                if item_id in item_df.index:
                    result.append(f'{item_id}: {item_df.loc[item_id]["item_attributes"]}')
                else:
                    logger.warning(f'Item {item_id} not found in item_df')
                    result.append(f'{item_id}: Unknown business')
            return result
        
        df['candidate_item_attributes'] = df['candidate_item_id'].apply(get_candidate_attributes)
        df['candidate_item_attributes'] = df['candidate_item_attributes'].apply(lambda x: '\n'.join(x))
    
    train_df, dev_df, test_df = dfs[0], dfs[1], dfs[2]
    all_df = pd.concat([train_df, dev_df, test_df])
    all_df = all_df.sort_values(by=['timestamp'], kind='mergesort')
    all_df = all_df.reset_index(drop=True)
    
    logger.info('Outputing data to csv files...')
    user_df.to_csv(os.path.join(dir, 'user.csv'))
    item_df.to_csv(os.path.join(dir, 'item.csv'))
    train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)
    all_df.to_csv(os.path.join(dir, 'all.csv'), index=False)
    logger.info('Successfully saved all CSV files')

if __name__ == '__main__':
    process_data(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'Yelp2018'), k_core=10)

