import os
import random
import pandas as pd
import numpy as np
import urllib.request
import gzip
import shutil
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.utils import append_his_info

def download_data(dir: str):
    raw_path = os.path.join(dir, 'raw_data')
    os.makedirs(raw_path, exist_ok=True)
    
    checkins_file = os.path.join(raw_path, 'loc-gowalla_totalCheckins.txt.gz')
    if not os.path.exists(checkins_file):
        logger.info('Downloading Gowalla check-ins dataset into ' + raw_path)
        try:
            url = 'https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz'
            urllib.request.urlretrieve(url, checkins_file)
            logger.info('Downloaded Gowalla check-ins successfully')
        except Exception as e:
            logger.error(f'Failed to download Gowalla data: {e}')
            raise
    
    # Extract the gz file
    checkins_txt = os.path.join(raw_path, 'loc-gowalla_totalCheckins.txt')
    if not os.path.exists(checkins_txt):
        logger.info('Extracting Gowalla check-ins data...')
        try:
            with gzip.open(checkins_file, 'rb') as f_in:
                with open(checkins_txt, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info('Extracted check-ins data successfully')
        except Exception as e:
            logger.error(f'Failed to extract Gowalla data: {e}')
            raise

def read_data(dir: str) -> pd.DataFrame:
    logger.info('Reading Gowalla check-ins file...')
    checkins_file = os.path.join(dir, 'loc-gowalla_totalCheckins.txt')
    
    # Read check-ins: [user] [check-in time] [latitude] [longitude] [location id]
    data_df = pd.read_csv(checkins_file, sep='\t', header=None, 
                          names=['user_id', 'timestamp', 'latitude', 'longitude', 'location_id'])
    
    logger.info(f'Successfully read {data_df.shape[0]} check-ins')
    return data_df

def process_user_data(data_df: pd.DataFrame) -> pd.DataFrame:
    # Create simple user profiles based on check-in statistics
    # Note: data_df should already have 'item_id' column (renamed from location_id)
    item_col = 'item_id' if 'item_id' in data_df.columns else 'location_id'
    user_stats = data_df.groupby('user_id').agg({
        item_col: 'count',
        'latitude': 'mean',
        'longitude': 'mean'
    }).rename(columns={item_col: 'checkin_count'})
    
    user_stats = user_stats.reset_index()
    user_stats = user_stats.set_index('user_id')
    
    template = PromptTemplate(
        template='User with {checkin_count} check-ins, centered around ({latitude:.2f}, {longitude:.2f})',
        input_variables=['checkin_count', 'latitude', 'longitude'],
    )
    user_stats['user_profile'] = user_stats.apply(lambda x: template.format(**x), axis=1)
    
    return user_stats

def process_item_data(data_df: pd.DataFrame) -> pd.DataFrame:
    # Create location (item) profiles based on coordinates
    # Note: data_df should already have 'item_id' column (renamed from location_id)
    item_col = 'item_id' if 'item_id' in data_df.columns else 'location_id'
    location_stats = data_df.groupby(item_col).agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'user_id': 'count'
    }).rename(columns={'user_id': 'visit_count'})
    
    location_stats = location_stats.reset_index()
    location_stats = location_stats.set_index(item_col)
    
    template = PromptTemplate(
        template='Location at ({latitude:.4f}, {longitude:.4f}) with {visit_count} visits',
        input_variables=['latitude', 'longitude', 'visit_count'],
    )
    location_stats['item_attributes'] = location_stats.apply(lambda x: template.format(**x), axis=1)
    
    return location_stats

def filter_data(data_df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """Apply k-core filtering to keep only users and items with >= min_interactions.
    
    Args:
        data_df: DataFrame with user-item interactions
        min_interactions: Minimum interactions required (default: 5 for 5-core)
    """
    original_size = data_df.shape[0]
    original_users = data_df['user_id'].nunique()
    original_items = data_df['location_id'].nunique()
    
    filter_before = -1
    while filter_before != data_df.shape[0]:
        filter_before = data_df.shape[0]
        data_df = data_df.groupby('user_id').filter(lambda x: len(x) >= min_interactions)
        data_df = data_df.groupby('location_id').filter(lambda x: len(x) >= min_interactions)
    
    filtered_size = data_df.shape[0]
    filtered_users = data_df['user_id'].nunique()
    filtered_items = data_df['location_id'].nunique()
    
    logger.info(f'{min_interactions}-core filtering: {original_size} -> {filtered_size} interactions')
    logger.info(f'Users: {original_users} -> {filtered_users}, Locations: {original_items} -> {filtered_items}')
    
    return data_df

def process_interaction_data(data_df: pd.DataFrame, n_neg_items: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Convert timestamp to datetime
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df['timestamp'] = data_df['timestamp'].astype(np.int64) // 10**9  # Convert to unix timestamp
    
    # Sort by timestamp
    data_df = data_df.sort_values(by=['timestamp'])
    data_df = filter_data(data_df)
    
    # Rename location_id to item_id for consistency
    data_df = data_df.rename(columns={'location_id': 'item_id'})
    
    # Create implicit rating (all check-ins = positive feedback)
    data_df['rating'] = 1
    
    # Build clicked item set
    clicked_item_set = {}
    for uid, group in data_df.groupby('user_id'):
        clicked_item_set[uid] = set(group['item_id'].values)
    
    # Get list of all valid item IDs
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

def process_data(dir: str, n_neg_items: int = 7):
    """Process the Gowalla raw data and output the processed data to `dir`.

    Args:
        dir (str): the directory of the dataset
        n_neg_items (int): Number of negative items to sample
    """
    logger.info(f'Starting to process Gowalla dataset in {dir}')
    
    download_data(dir)
    raw_data_dir = os.path.join(dir, "raw_data")
    
    data_df = read_data(raw_data_dir)
    
    logger.info('Processing interaction data...')
    train_df, dev_df, test_df, filtered_data_df = process_interaction_data(data_df, n_neg_items)
    
    logger.info('Processing user data...')
    user_df = process_user_data(filtered_data_df)
    logger.info(f'Number of users: {user_df.shape[0]}')
    
    logger.info('Processing location data...')
    item_df = process_item_data(filtered_data_df)
    logger.info(f'Number of locations: {item_df.shape[0]}')
    
    logger.info('Appending history information...')
    dfs = append_his_info([train_df, dev_df, test_df], neg=True)
    
    for df in dfs:
        df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
        df['history'] = df.apply(lambda x: [f'{item_attributes}' for item_attributes in x['history']], axis=1)
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
                    result.append(f'{item_id}: Unknown location')
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
    process_data(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'Gowalla'))

