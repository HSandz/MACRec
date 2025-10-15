import os
import random
import subprocess
import pandas as pd
import numpy as np
import platform
import urllib.request
import zipfile
import shutil
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
            url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
            urllib.request.urlretrieve(url, zip_file_path)
            logger.info('Downloaded ml-100k.zip successfully')
        except Exception as e:
            logger.error(f'Failed to download ml-100k.zip: {e}')
            raise
    
    if not os.path.exists(os.path.join(raw_path, 'u.data')):
        logger.info('Unzipping ml-100k dataset into ' + raw_path)
        try:
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
    """Read raw data files from directory.
    
    Args:
        dir: Directory containing raw data files
        
    Returns:
        Tuple of (data_df, item_df, user_df, genre_df)
    """
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
        
        # Validate that data was read correctly
        if data_df.empty or item_df.empty or user_df.empty or genre_df.empty:
            raise ValueError("One or more data files are empty")
        
        logger.info(f'Successfully read data files: {data_df.shape[0]} interactions, {item_df.shape[0]} items, {user_df.shape[0]} users, {genre_df.shape[0]} genres')
        return data_df, item_df, user_df, genre_df
        
    except FileNotFoundError as e:
        logger.error(f'Required data file not found: {e}')
        raise
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

def filter_data(data_df: pd.DataFrame, min_interactions: int = 5, max_iterations: int = 10) -> pd.DataFrame:
    """Filter data to retain only users and items with at least min_interactions.
    
    Args:
        data_df: DataFrame with user-item interactions
        min_interactions: Minimum number of interactions required
        max_iterations: Maximum number of filter iterations to prevent infinite loop
    
    Returns:
        Filtered DataFrame
    """
    original_size = data_df.shape[0]
    iteration = 0
    filter_before = -1
    
    while filter_before != data_df.shape[0] and iteration < max_iterations:
        filter_before = data_df.shape[0]
        
        # Filter users with at least min_interactions
        user_counts = data_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        data_df = data_df[data_df['user_id'].isin(valid_users)]
        
        # Filter items with at least min_interactions
        item_counts = data_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        data_df = data_df[data_df['item_id'].isin(valid_items)]
        
        iteration += 1
    
    if iteration >= max_iterations:
        logger.warning(f'Filter reached max iterations ({max_iterations}). Some users/items may have < {min_interactions} interactions.')
    
    filtered_size = data_df.shape[0]
    logger.info(f'Filtered data: {original_size} -> {filtered_size} interactions ({filtered_size/original_size*100:.1f}% retained)')
    
    return data_df

def process_interaction_data(data_df: pd.DataFrame, n_neg_items: int = 9) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    # sort data_df by timestamp
    data_df = data_df.sort_values(by=['timestamp'])
    data_df = filter_data(data_df)
    
    if data_df.empty:
        raise ValueError("No data remains after filtering. Consider reducing min_interactions parameter.")
    
    clicked_item_set = dict()
    for user_id, seq_df in data_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())

    n_items = data_df['item_id'].nunique()
    min_item_id = data_df['item_id'].min()
    max_item_id = data_df['item_id'].max()

    def negative_sample(df):
        neg_items = np.random.randint(min_item_id, max_item_id + 1, (len(df), n_neg_items))
        for i, uid in enumerate(df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                attempts = 0
                while (neg_items[i][j] in user_clicked or 
                       neg_items[i][j] in neg_items[i][:j]) and attempts < 100:
                    neg_items[i][j] = np.random.randint(min_item_id, max_item_id + 1)
                    attempts += 1
                if attempts >= 100:
                    logger.warning(f"Could not find unique negative item for user {uid} after 100 attempts")
            # Verify uniqueness
            if len(set(neg_items[i])) != len(neg_items[i]):
                logger.warning(f"Duplicate negative items found for user {uid}")
        df['neg_item_id'] = neg_items.tolist()
        return df

    def generate_dev_test(data_df: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        """Generate dev and test sets by taking the last 2 interactions per user"""
        result_dfs = []
        for idx in range(2):  # Take last 2 interactions for test and dev
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df

    data_df = negative_sample(data_df)
    # Keep the first interaction of each user for training (ensures each user has training data)
    keep_first_df = data_df.groupby('user_id').head(1)
    remaining_df = data_df.drop(keep_first_df.index)

    [test_df, dev_df], train_remaining_df = generate_dev_test(remaining_df)
    train_df = pd.concat([keep_first_df, train_remaining_df]).sort_index()
    
    # Verify data split
    logger.info(f'Data split - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}')
    logger.info(f'Unique users - Train: {train_df["user_id"].nunique()}, Dev: {dev_df["user_id"].nunique()}, Test: {test_df["user_id"].nunique()}')
    
    return train_df, dev_df, test_df

def process_interaction_data_with_retrieval(data_df: pd.DataFrame, n_retrieval_items: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process interaction data using retrieval-based sampling instead of negative sampling."""
    data_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    # sort data_df by timestamp
    data_df = data_df.sort_values(by=['timestamp'])
    data_df = filter_data(data_df)
    
    if data_df.empty:
        raise ValueError("No data remains after filtering. Consider reducing min_interactions parameter.")
    
    # Ensure mapping files exist for embedding retriever
    from macrec.utils.mapping_utils import ensure_embedding_mappings
    embeddings_dir = "data/ml-100k/embeddings"
    if not ensure_embedding_mappings(embeddings_dir, "ml-100k"):
        logger.warning("Failed to create mapping files for ml-100k embeddings")
    
    # Load embedding retriever
    from macrec.tools.embedding_retriever import EmbeddingRetriever
    embedding_retriever = EmbeddingRetriever(config_path="config/tools/embedding_retriever.json")
    
    def retrieval_sample(df):
        """Use embedding-based retrieval to get top-K similar items for each user."""
        candidate_items = []
        for _, row in df.iterrows():
            user_id = row['user_id']
            target_item = row['item_id']
            
            try:
                # Get top-K items from embedding retriever (pure similarity-based, no positive item added)
                top_items, _ = embedding_retriever.retrieve(user_id=user_id, k=n_retrieval_items)
                
                # Use only the top-K similar items, do NOT add target item
                candidate_items.append(top_items)
            except Exception as e:
                logger.warning(f"Failed to retrieve items for user {user_id}: {e}")
                # Fallback to random items or empty list
                candidate_items.append([])
        
        df['candidate_item_id'] = candidate_items
        return df

    def generate_dev_test(data_df: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        """Generate dev and test sets by taking the last 2 interactions per user"""
        result_dfs = []
        for idx in range(2):  # Take last 2 interactions for test and dev
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df

    data_df = retrieval_sample(data_df)
    # Keep the first interaction of each user for training (ensures each user has training data)
    keep_first_df = data_df.groupby('user_id').head(1)
    remaining_df = data_df.drop(keep_first_df.index)

    [test_df, dev_df], train_remaining_df = generate_dev_test(remaining_df)
    train_df = pd.concat([keep_first_df, train_remaining_df]).sort_index()
    
    # Verify data split
    logger.info(f'Data split - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}')
    logger.info(f'Unique users - Train: {train_df["user_id"].nunique()}, Dev: {dev_df["user_id"].nunique()}, Test: {test_df["user_id"].nunique()}')
    
    return train_df, dev_df, test_df

def process_data(dir: str, n_neg_items: int = 9):
    """Process the ml-100k raw data and output the processed data to `dir`.

    Args:
        `dir (str)`: the directory of the dataset. We suppose the raw data is in `dir/raw_data` and the processed data will be output to `dir`.
        `n_neg_items (int)`: Number of negative items to sample for each positive interaction.
    """
    try:
        logger.info(f'Starting to process ml-100k dataset in {dir}')
        
        download_data(dir)
        raw_data_dir = os.path.join(dir, "raw_data")
        
        logger.info('Reading raw data files...')
        data_df, item_df, user_df, genre_df = read_data(raw_data_dir)
        
        logger.info('Processing user data...')
        user_df = process_user_data(user_df)
        logger.info(f'Number of users: {user_df.shape[0]}')
        
        logger.info('Processing item data...')
        item_df = process_item_data(item_df)
        logger.info(f'Number of items: {item_df.shape[0]}')
        
        logger.info('Processing interaction data...')
        train_df, dev_df, test_df = process_interaction_data(data_df, n_neg_items)
        logger.info(f'Number of train interactions: {train_df.shape[0]}')
        logger.info(f'Number of dev interactions: {dev_df.shape[0]}')
        logger.info(f'Number of test interactions: {test_df.shape[0]}')
        
        logger.info('Appending history information...')
        dfs = append_his_info([train_df, dev_df, test_df], neg=True)
        logger.info('Completed append history information to interactions')
        
        logger.info('Processing history and user profiles...')
        for i, df in enumerate(dfs):
            df_name = ['train', 'dev', 'test'][i]
            logger.info(f'Processing {df_name} set...')
            
            # format history by list the historical item attributes
            df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
            # concat the attributes with item's rating
            df['history'] = df.apply(lambda x: [f'{item_attributes} (rating: {rating})' for item_attributes, rating in zip(x['history'], x['history_rating'])], axis=1)
            # Separate each item attributes by a newline
            df['history'] = df['history'].apply(lambda x: '\n'.join(x))
            # add user profile for this interaction
            df['user_profile'] = df['user_id'].apply(lambda x: user_df.loc[x]['user_profile'])
            df['target_item_attributes'] = df['item_id'].apply(lambda x: item_df.loc[x]['item_attributes'])
            # candidates id
            df['candidate_item_id'] = df.apply(lambda x: [x['item_id']]+x['neg_item_id'], axis=1)

            # Shuffle candidate items in-place
            df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: random.sample(x, len(x)))

            # add item attributes
            def get_candidate_attributes(candidate_ids):
                """Get formatted candidate item attributes"""
                try:
                    candidate_item_attributes = []
                    for item_id in candidate_ids:
                        if item_id in item_df.index:
                            item_attributes = item_df.loc[item_id]['item_attributes']
                            candidate_item_attributes.append(f'{item_id}: {item_attributes}')
                        else:
                            logger.warning(f'Item {item_id} not found in item_df')
                            candidate_item_attributes.append(f'{item_id}: Unknown item')
                    return candidate_item_attributes
                except Exception as e:
                    logger.error(f'Error processing candidate attributes: {e}')
                    return [f'{item_id}: Error' for item_id in candidate_ids]

            df['candidate_item_attributes'] = df['candidate_item_id'].apply(get_candidate_attributes)
            df['candidate_item_attributes'] = df['candidate_item_attributes'].apply(lambda x: '\n'.join(x))
            
            # replace empty string with 'None'
            for col in df.columns.to_list():
                df[col] = df[col].apply(lambda x: 'None' if x == '' else x)

        train_df = dfs[0]
        dev_df = dfs[1]
        test_df = dfs[2]
        all_df = pd.concat([train_df, dev_df, test_df])
        all_df = all_df.sort_values(by=['timestamp'], kind='mergesort')
        all_df = all_df.reset_index(drop=True)
        
        logger.info('Outputing data to csv files...')
        try:
            user_df.to_csv(os.path.join(dir, 'user.csv'))
            item_df.to_csv(os.path.join(dir, 'item.csv'))
            train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
            dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
            test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)
            all_df.to_csv(os.path.join(dir, 'all.csv'), index=False)
            logger.info('Successfully saved all CSV files')
        except Exception as e:
            logger.error(f'Error saving CSV files: {e}')
            raise
            
    except Exception as e:
        logger.error(f'Error processing ml-100k dataset: {e}')
        raise

def process_data_with_retrieval(dir: str, n_retrieval_items: int = 20):
    """Process the ml-100k raw data using retrieval-based sampling and output the processed data to `dir`.

    Args:
        `dir (str)`: the directory of the dataset. We suppose the raw data is in `dir/raw_data` and the processed data will be output to `dir`.
        `n_retrieval_items (int)`: Number of retrieval items to get for each positive interaction.
    """
    try:
        logger.info(f'Starting to process ml-100k dataset with retrieval-based sampling in {dir}')
        
        download_data(dir)
        raw_data_dir = os.path.join(dir, "raw_data")
        
        logger.info('Reading raw data files...')
        data_df, item_df, user_df, genre_df = read_data(raw_data_dir)
        
        logger.info('Processing user data...')
        user_df = process_user_data(user_df)
        logger.info(f'Number of users: {user_df.shape[0]}')
        
        logger.info('Processing item data...')
        item_df = process_item_data(item_df)
        logger.info(f'Number of items: {item_df.shape[0]}')
        
        logger.info('Processing interaction data with retrieval-based sampling...')
        train_df, dev_df, test_df = process_interaction_data_with_retrieval(data_df, n_retrieval_items)
        logger.info(f'Number of train interactions: {train_df.shape[0]}')
        logger.info(f'Number of dev interactions: {dev_df.shape[0]}')
        logger.info(f'Number of test interactions: {test_df.shape[0]}')
        
        logger.info('Appending history information...')
        dfs = append_his_info([train_df, dev_df, test_df], neg=False)  # neg=False since we don't have neg_item_id
        logger.info('Completed append history information to interactions')
        
        logger.info('Processing history and user profiles...')
        for i, df in enumerate(dfs):
            df_name = ['train', 'dev', 'test'][i]
            logger.info(f'Processing {df_name} set...')
            
            # format history by list the historical item attributes
            df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
            # concat the attributes with item's rating
            df['history'] = df.apply(lambda x: [f'{item_attributes} (rating: {rating})' for item_attributes, rating in zip(x['history'], x['history_rating'])], axis=1)
            # Separate each item attributes by a newline
            df['history'] = df['history'].apply(lambda x: '\n'.join(x))
            # add user profile for this interaction
            df['user_profile'] = df['user_id'].apply(lambda x: user_df.loc[x]['user_profile'])
            df['target_item_attributes'] = df['item_id'].apply(lambda x: item_df.loc[x]['item_attributes'])
            
            # Shuffle candidate items in-place (candidate_item_id already set in retrieval_sample)
            if 'candidate_item_id' in df.columns:
                df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: random.sample(x, len(x)) if len(x) > 0 else [])
            else:
                # Fallback: create empty candidate_item_id column if it doesn't exist
                df['candidate_item_id'] = [[] for _ in range(len(df))]

            # add item attributes
            def get_candidate_attributes(candidate_ids):
                """Get formatted candidate item attributes"""
                try:
                    candidate_item_attributes = []
                    for item_id in candidate_ids:
                        if item_id in item_df.index:
                            item_attributes = item_df.loc[item_id]['item_attributes']
                            candidate_item_attributes.append(f'{item_id}: {item_attributes}')
                        else:
                            logger.warning(f'Item {item_id} not found in item_df')
                            candidate_item_attributes.append(f'{item_id}: Unknown item')
                    return candidate_item_attributes
                except Exception as e:
                    logger.error(f'Error processing candidate attributes: {e}')
                    return [f'{item_id}: Error' for item_id in candidate_ids]

            df['candidate_item_attributes'] = df['candidate_item_id'].apply(get_candidate_attributes)
            df['candidate_item_attributes'] = df['candidate_item_attributes'].apply(lambda x: '\n'.join(x))
            
            # replace empty string with 'None'
            for col in df.columns.to_list():
                df[col] = df[col].apply(lambda x: 'None' if x == '' else x)

        train_df = dfs[0]
        dev_df = dfs[1]
        test_df = dfs[2]
        all_df = pd.concat([train_df, dev_df, test_df])
        all_df = all_df.sort_values(by=['timestamp'], kind='mergesort')
        all_df = all_df.reset_index(drop=True)
        
        logger.info('Outputing data to csv files...')
        try:
            user_df.to_csv(os.path.join(dir, 'user.csv'))
            item_df.to_csv(os.path.join(dir, 'item.csv'))
            train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
            dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
            test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)
            all_df.to_csv(os.path.join(dir, 'all.csv'), index=False)
            logger.info('Successfully saved all CSV files')
        except Exception as e:
            logger.error(f'Error saving CSV files: {e}')
            raise
            
    except Exception as e:
        logger.error(f'Error processing ml-100k dataset with retrieval: {e}')
        raise

if __name__ == '__main__':
    process_data(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ml-100k'))