import os
import random
import subprocess
import pandas as pd
import numpy as np
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.utils import append_his_info

def download_data(dir: str):
    raw_path = os.path.join(dir, 'raw_data')
    os.makedirs(raw_path, exist_ok=True)
    
    zip_file_path = os.path.join(raw_path, 'ml-100k.zip')
    if not os.path.exists(zip_file_path):
        logger.info('Downloading ml-100k dataset into ' + raw_path)
        try:
            import urllib.request
            url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
            urllib.request.urlretrieve(url, zip_file_path)
            logger.info('Downloaded ml-100k.zip successfully')
        except Exception as e:
            logger.error(f'Failed to download ml-100k.zip: {e}')
            raise
    
    if not os.path.exists(os.path.join(raw_path, 'u.data')):
        logger.info('Unzipping ml-100k dataset into ' + raw_path)
        try:
            import zipfile
            import shutil
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(raw_path)
            
            # Move files from ml-100k subdirectory to raw_path
            ml100k_dir = os.path.join(raw_path, 'ml-100k')
            if os.path.exists(ml100k_dir):
                for filename in os.listdir(ml100k_dir):
                    src = os.path.join(ml100k_dir, filename)
                    dst = os.path.join(raw_path, filename)
                    shutil.move(src, dst)
                os.rmdir(ml100k_dir)
            logger.info('Extracted and organized files successfully')
        except Exception as e:
            logger.error(f'Failed to extract ml-100k.zip: {e}')
            raise

def read_data(dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read raw data files from directory."""
    try:
        logger.info('Reading u.data file...')
        with open(os.path.join(dir, 'u.data'), 'r') as f:
            data_df = pd.read_csv(f, sep='\t', header=None)
        
        logger.info('Reading u.item file...')
        with open(os.path.join(dir, 'u.item'), 'r', encoding='ISO-8859-1') as f:
            item_df = pd.read_csv(f, sep='|', header=None, encoding='ISO-8859-1')
        
        logger.info('Reading u.user file...')
        with open(os.path.join(dir, 'u.user'), 'r') as f:
            user_df = pd.read_csv(f, sep='|', header=None)
        
        logger.info('Reading u.genre file...')
        with open(os.path.join(dir, 'u.genre'), 'r') as f:
            genre_df = pd.read_csv(f, sep='|', header=None)
        
        logger.info(f'Successfully read data files: {data_df.shape[0]} interactions, {item_df.shape[0]} items, {user_df.shape[0]} users, {genre_df.shape[0]} genres')
        return data_df, item_df, user_df, genre_df
        
    except Exception as e:
        logger.error(f'Error reading data files: {e}')
        raise

def process_user_data(user_df: pd.DataFrame) -> pd.DataFrame:
    user_df.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_df = user_df.drop(columns=['zip_code'])
    user_df = user_df.set_index('user_id')
    # Update M and F into male and female
    user_df['gender'] = user_df['gender'].apply(lambda x: 'male' if x == 'M' else 'female')
    # set a user profile column to be a string contain all the user information with a template
    input_variables = user_df.columns.to_list()
    template = PromptTemplate(
        template='Age: {age}\nGender: {gender}\nOccupation: {occupation}',
        input_variables=input_variables,
    )
    user_df['user_profile'] = user_df[input_variables].apply(lambda x: template.format(**x), axis=1)

    for col in user_df.columns.to_list():
        user_df[col] = user_df[col].apply(lambda x: 'None' if x == '' else x)
    return user_df

def process_item_data(item_df: pd.DataFrame) -> pd.DataFrame:
    item_df.columns = ['item_id', 'title', 'release_date', 'video_release_date',
                       'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                       'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genres = item_df.columns.to_list()[5:]
    item_df = item_df.drop(columns=['IMDb_URL'])
    item_df = item_df.set_index('item_id')
    # set video_release_date to unknown if it is null
    item_df['video_release_date'] = item_df['video_release_date'].fillna('unknown')
    # set release_date to unknown if it is null
    item_df['release_date'] = item_df['release_date'].fillna('unknown')

    # set a genre column to be a list of genres
    def get_genre(x: pd.Series) -> list[str]:
        return '|'.join([genre for genre, value in x.items() if value == 1])

    item_df['genre'] = item_df[genres].apply(lambda x: get_genre(x), axis=1)
    # set a item_attributes column to be a string contain all the item information with a template
    input_variables = item_df.columns.to_list()[:3] + ['genre']
    template = PromptTemplate(
        template='Title: {title}, Genres: {genre}',
        input_variables=input_variables,
    )
    item_df['item_attributes'] = item_df[input_variables].apply(lambda x: template.format(**x), axis=1)
    # drop original genre columns
    item_df = item_df.drop(columns=genres)
    return item_df

def filter_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to retain only users and items with at least 5 interactions."""
    filter_before = -1
    while filter_before != data_df.shape[0]:
        filter_before = data_df.shape[0]
        data_df = data_df.groupby('user_id').filter(lambda x: len(x) >= 5)
        data_df = data_df.groupby('item_id').filter(lambda x: len(x) >= 5)
    return data_df

def process_interaction_data_negative(data_df: pd.DataFrame, n_neg_items: int = 9) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process interaction data using negative sampling strategy."""
    data_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    data_df = data_df.sort_values(by=['timestamp'])
    data_df = filter_data(data_df)
    
    clicked_item_set = dict()
    for user_id, seq_df in data_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())

    n_items = data_df['item_id'].nunique()

    def negative_sample(df):
        neg_items = np.random.randint(1, n_items + 1, (len(df), n_neg_items))
        for i, uid in enumerate(df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                attempts = 0
                while (neg_items[i][j] in user_clicked or 
                       neg_items[i][j] in neg_items[i][:j]) and attempts < 100:
                    neg_items[i][j] = np.random.randint(1, n_items + 1)
                    attempts += 1
                if attempts >= 100:
                    logger.warning(f"Could not find unique negative item for user {uid} after 100 attempts")
        df['neg_item_id'] = neg_items.tolist()
        return df

    def generate_dev_test(data_df: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df

    data_df = negative_sample(data_df)
    leave_df = data_df.groupby('user_id').head(1)
    left_df = data_df.drop(leave_df.index)

    [test_df, dev_df], train_df = generate_dev_test(left_df)
    train_df = pd.concat([leave_df, train_df]).sort_index()
    return train_df, dev_df, test_df

def process_interaction_data_retrieval(data_df: pd.DataFrame, n_retrieval_items: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process interaction data using retrieval strategy (top-rated items)."""
    data_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    data_df = data_df.sort_values(by=['timestamp'])
    data_df = filter_data(data_df)
    
    # Calculate average ratings for each item
    item_ratings = data_df.groupby('item_id')['rating'].mean().sort_values(ascending=False)
    top_rated_items = item_ratings.head(n_retrieval_items * 2).index.tolist()  # Get more items to ensure enough after filtering
    
    clicked_item_set = dict()
    for user_id, seq_df in data_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())

    def retrieval_sample(df):
        """Sample top-rated items excluding positive items."""
        candidate_items = []
        for _, row in df.iterrows():
            user_id = row['user_id']
            positive_item = row['item_id']
            user_clicked = clicked_item_set[user_id]
            
            # Get top-rated items excluding positive and user's clicked items
            available_items = [item for item in top_rated_items 
                             if item not in user_clicked and item != positive_item]
            
            # Take top n_retrieval_items
            sampled_items = available_items[:n_retrieval_items]
            candidate_items.append(sampled_items)
        
        df['candidate_item_id'] = candidate_items
        return df

    def generate_dev_test(data_df: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df

    data_df = retrieval_sample(data_df)
    leave_df = data_df.groupby('user_id').head(1)
    left_df = data_df.drop(leave_df.index)

    [test_df, dev_df], train_df = generate_dev_test(left_df)
    train_df = pd.concat([leave_df, train_df]).sort_index()
    return train_df, dev_df, test_df

def _process_dataframes(dfs: list[pd.DataFrame], item_df: pd.DataFrame, user_df: pd.DataFrame, strategy: str):
    """Process dataframes with history and candidate information."""
    neg_flag = (strategy == 'negative')
    dfs = append_his_info(dfs, neg=neg_flag)
    
    for df in dfs:
        # Format history
        df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
        df['history'] = df.apply(lambda x: [f'{item_attributes} (rating: {rating})' for item_attributes, rating in zip(x['history'], x['history_rating'])], axis=1)
        df['history'] = df['history'].apply(lambda x: '\n'.join(x))
        
        # Add user profile and target item attributes
        df['user_profile'] = df['user_id'].apply(lambda x: user_df.loc[x]['user_profile'])
        df['target_item_attributes'] = df['item_id'].apply(lambda x: item_df.loc[x]['item_attributes'])
        
        # Prepare candidate items based on strategy
        if strategy == 'negative':
            df['candidate_item_id'] = df.apply(lambda x: [x['item_id']] + x['neg_item_id'], axis=1)
        else:  # retrieval strategy
            # candidate_item_id is already set in retrieval_sample
            pass
        
        # Shuffle candidate items
        def shuffle_candidates(x):
            random.shuffle(x)
            return x
        if 'candidate_item_id' in df.columns:
            df['candidate_item_id'] = df['candidate_item_id'].apply(shuffle_candidates)

        # Add candidate item attributes
        def candidate_attr(x):
            candidate_item_attributes = []
            for item_id in x:
                if item_id in item_df.index:
                    item_attributes = item_df.loc[item_id]['item_attributes']
                    candidate_item_attributes.append(f'{item_id}: {item_attributes}')
                else:
                    candidate_item_attributes.append(f'{item_id}: Unknown item')
            return candidate_item_attributes

        df['candidate_item_attributes'] = df['candidate_item_id'].apply(candidate_attr)
        df['candidate_item_attributes'] = df['candidate_item_attributes'].apply(lambda x: '\n'.join(x))
        
        # Replace empty strings with 'None'
        for col in df.columns.to_list():
            df[col] = df[col].apply(lambda x: 'None' if x == '' else x)

    return dfs

def process_data(dir: str, n_neg_items: int = 9, sampling_strategy: str = 'negative', n_retrieval_items: int = 20):
    """Process ml-100k dataset with specified sampling strategy.
    
    Args:
        dir: Directory of the dataset
        n_neg_items: Number of negative items (for negative sampling)
        sampling_strategy: 'negative' or 'retrieval'
        n_retrieval_items: Number of retrieval items (for retrieval sampling)
    """
    logger.info(f'Processing ml-100k dataset in {dir} with {sampling_strategy} strategy')
    
    download_data(dir)
    data_df, item_df, user_df, genre_df = read_data(os.path.join(dir, "raw_data"))
    user_df = process_user_data(user_df)
    item_df = process_item_data(item_df)
    
    logger.info(f'Number of users: {user_df.shape[0]}')
    logger.info(f'Number of items: {item_df.shape[0]}')
    
    # Process interaction data based on strategy
    if sampling_strategy == 'retrieval':
        train_df, dev_df, test_df = process_interaction_data_retrieval(data_df, n_retrieval_items)
    else:  # negative sampling
        train_df, dev_df, test_df = process_interaction_data_negative(data_df, n_neg_items)
    
    logger.info(f'Number of train interactions: {train_df.shape[0]}')
    logger.info(f'Number of dev interactions: {dev_df.shape[0]}')
    logger.info(f'Number of test interactions: {test_df.shape[0]}')
    
    # Process dataframes
    dfs = _process_dataframes([train_df, dev_df, test_df], item_df, user_df, sampling_strategy)
    
    # Save files
    train_df, dev_df, test_df = dfs
    all_df = pd.concat([train_df, dev_df, test_df])
    all_df = all_df.sort_values(by=['timestamp'], kind='mergesort')
    all_df = all_df.reset_index(drop=True)
    
    logger.info('Outputting data to CSV files...')
    user_df.to_csv(os.path.join(dir, 'user.csv'))
    item_df.to_csv(os.path.join(dir, 'item.csv'))
    train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)
    all_df.to_csv(os.path.join(dir, 'all.csv'), index=False)
    logger.info('Successfully saved all CSV files')

# Backward compatibility functions
def process_data_with_retrieval(dir: str, n_retrieval_items: int = 20):
    """Process ml-100k with retrieval sampling (backward compatibility)."""
    return process_data(dir, n_neg_items=9, sampling_strategy='retrieval', n_retrieval_items=n_retrieval_items)

if __name__ == '__main__':
    process_data(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ml-100k'))