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
    if not os.path.exists(os.path.join(raw_path, 'ml-1m.zip')):
        logger.info('Downloading ml-1m dataset into ' + raw_path)
        if os.name == 'nt':  # Windows
            subprocess.call(
                f'cd /d "{raw_path}" && curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip', 
                shell=True)
        else:  # Unix/Linux/Mac
            subprocess.call(
                f'cd {raw_path} && curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip',
                shell=True)
    if not os.path.exists(os.path.join(raw_path, 'ratings.dat')):
        logger.info('Unzipping ml-1m dataset into ' + raw_path)
        # Prefer Python's zipfile extraction (portable), fall back to system tools if necessary
        try:
            import zipfile
            with zipfile.ZipFile(os.path.join(raw_path, 'ml-1m.zip'), 'r') as zf:
                zf.extractall(raw_path)
        except Exception as e:
            logger.warning(f'zipfile extraction failed ({e}), falling back to system commands')
            if os.name == 'nt':  # Windows
                subprocess.call(
                    f'powershell -Command "Expand-Archive -Path \\"{os.path.join(raw_path, "ml-1m.zip")}\\" -DestinationPath \\"{raw_path}\\" -Force"', 
                    shell=True)
                # Move files from ml-1m subfolder to raw_data
                ml1m_path = os.path.join(raw_path, 'ml-1m')
                if os.path.exists(ml1m_path):
                    for file in os.listdir(ml1m_path):
                        src = os.path.join(ml1m_path, file)
                        dst = os.path.join(raw_path, file)
                        if os.path.isfile(src):
                            os.rename(src, dst)
                    os.rmdir(ml1m_path)
            else:  # Unix/Linux/Mac
                subprocess.call(
                    f'cd {raw_path} && unzip -o ml-1m.zip', shell=True)
            # If extraction created ml-1m subfolder, move files to raw_data
            ml1m_path = os.path.join(raw_path, 'ml-1m')
            if os.path.exists(ml1m_path):
                for file in os.listdir(ml1m_path):
                    src = os.path.join(ml1m_path, file)
                    dst = os.path.join(raw_path, file)
                    if os.path.isfile(src):
                        os.rename(src, dst)
                os.rmdir(ml1m_path)

def read_data(dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # MovieLens 1M uses :: as separator
    # Use latin-1 encoding for files that may contain non-UTF-8 characters (e.g. movie titles)
    ratings_df = pd.read_csv(os.path.join(dir, 'ratings.dat'), sep='::', header=None, engine='python', encoding='latin-1')
    movies_df = pd.read_csv(os.path.join(dir, 'movies.dat'), sep='::', header=None, engine='python', encoding='latin-1')
    users_df = pd.read_csv(os.path.join(dir, 'users.dat'), sep='::', header=None, engine='python', encoding='latin-1')
    return ratings_df, movies_df, users_df

def process_user_data(user_df: pd.DataFrame) -> pd.DataFrame:
    user_df.columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    user_df = user_df.drop(columns=['zip_code'])
    user_df = user_df.set_index('user_id')
    
    # Update M and F into male and female
    user_df['gender'] = user_df['gender'].apply(lambda x: 'male' if x == 'M' else 'female')
    
    # Map age groups to readable format
    age_mapping = {
        1: "Under 18",
        18: "18-24", 
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"
    }
    user_df['age'] = user_df['age'].map(age_mapping)
    
    # Map occupation codes to names
    occupation_mapping = {
        0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
        4: "college/grad student", 5: "customer service", 6: "doctor/health care",
        7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
        11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
        15: "scientist", 16: "self-employed", 17: "technician/engineer",
        18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
    }
    user_df['occupation'] = user_df['occupation'].map(occupation_mapping)
    
    # Set user profile template
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
    item_df.columns = ['item_id', 'title', 'genres']
    item_df = item_df.set_index('item_id')
    
    # Parse genres
    item_df['genre'] = item_df['genres'].apply(lambda x: x.split('|'))
    
    # Extract year from title
    item_df['year'] = item_df['title'].str.extract(r'\((\d{4})\)$')
    item_df['title'] = item_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
    
    # Create item attributes template
    input_variables = ['title', 'genre', 'year']
    template = PromptTemplate(
        template='Title: {title}, Genres: {genre}, Year: {year}',
        input_variables=input_variables,
    )
    item_df['item_attributes'] = item_df[input_variables].apply(
        lambda x: template.format(
            title=x['title'],
            genre='|'.join(x['genre']) if isinstance(x['genre'], list) else x['genre'],
            year=x['year'] if pd.notna(x['year']) else 'Unknown'
        ), axis=1
    )
    
    # Drop original genres column
    item_df = item_df.drop(columns=['genres'])
    return item_df

def filter_data(data_df: pd.DataFrame) -> pd.DataFrame:
    # Filter data_df, only retain users and items with at least 5 associated interactions
    filter_before = -1
    while filter_before != data_df.shape[0]:
        filter_before = data_df.shape[0]
        data_df = data_df.groupby('user_id').filter(lambda x: len(x) >= 5)
        data_df = data_df.groupby('item_id').filter(lambda x: len(x) >= 5)
    return data_df

def process_interaction_data(data_df: pd.DataFrame, n_neg_items: int = 9) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    # Sort data_df by timestamp
    data_df = data_df.sort_values(by=['timestamp'])
    data_df = filter_data(data_df)
    
    # Get number of unique items and create clicked item set
    n_items = data_df['item_id'].max()
    clicked_item_set = {}
    for uid, group in data_df.groupby('user_id'):
        clicked_item_set[uid] = set(group['item_id'].values)
    
    def generate_dev_test(data_df: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df
    
    def negative_sample(df):
        neg_items = np.random.randint(1, n_items + 1, (len(df), n_neg_items))
        for i, uid in enumerate(df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                while neg_items[i][j] in user_clicked or neg_items[i][j] in neg_items[i][:j]:
                    neg_items[i][j] = np.random.randint(1, n_items + 1)
            assert len(set(neg_items[i])) == len(neg_items[i])  # check if there is duplicate item id
        df['neg_item_id'] = neg_items.tolist()
        return df

    # Split data into train/dev/test
    leave_df = data_df.groupby('user_id').head(1)
    left_df = data_df.drop(leave_df.index)

    [test_df, dev_df], train_df = generate_dev_test(left_df)
    train_df = pd.concat([leave_df, train_df]).sort_index()
    
    # Apply negative sampling to all splits
    train_df = negative_sample(train_df)
    dev_df = negative_sample(dev_df)
    test_df = negative_sample(test_df)
    
    return train_df, dev_df, test_df

def process_interaction_data_with_retrieval(data_df: pd.DataFrame, n_retrieval_items: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process interaction data using retrieval-based sampling instead of negative sampling."""
    data_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    # Sort data_df by timestamp
    data_df = data_df.sort_values(by=['timestamp'])
    data_df = filter_data(data_df)
    
    # Ensure mapping files exist for embedding retriever
    from macrec.utils.mapping_utils import ensure_embedding_mappings
    embeddings_dir = "data/ml-1m/embeddings"
    if not ensure_embedding_mappings(embeddings_dir, "ml-1m"):
        logger.warning("Failed to create mapping files for ml-1m embeddings")
    
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
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df

    # Split data into train/dev/test
    leave_df = data_df.groupby('user_id').head(1)
    left_df = data_df.drop(leave_df.index)

    [test_df, dev_df], train_df = generate_dev_test(left_df)
    train_df = pd.concat([leave_df, train_df]).sort_index()
    
    # Apply retrieval sampling to all splits
    train_df = retrieval_sample(train_df)
    dev_df = retrieval_sample(dev_df)
    test_df = retrieval_sample(test_df)
    
    return train_df, dev_df, test_df

def process_data(dir: str, n_neg_items: int = 9):
    """Process the ml-1m raw data and output the processed data to `dir`.

    Args:
        `dir (str)`: the directory of the dataset. We suppose the raw data is in `dir/raw_data` and the processed data will be output to `dir`.
    """
    download_data(dir)
    data_df, item_df, user_df = read_data(os.path.join(dir, "raw_data"))
    user_df = process_user_data(user_df)
    logger.info(f'Number of users: {user_df.shape[0]}')
    item_df = process_item_data(item_df)
    logger.info(f'Number of items: {item_df.shape[0]}')
    train_df, dev_df, test_df = process_interaction_data(data_df, n_neg_items)
    logger.info(f'Number of train interactions: {train_df.shape[0]}')
    dfs = append_his_info([train_df, dev_df, test_df], neg=True)
    logger.info('Completed append history information to interactions')

    for df in dfs:
        # Format history by listing the historical item attributes
        df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
        # Concat the attributes with item's rating
        df['history'] = df.apply(lambda x: [f'{item_attributes} (rating: {rating})' for item_attributes, rating in zip(x['history'], x['history_rating'])], axis=1)
        # Separate each item attributes by a newline
        df['history'] = df['history'].apply(lambda x: '\n'.join(x))
        # Add user profile for this interaction
        df['user_profile'] = df['user_id'].apply(lambda x: user_df.loc[x]['user_profile'])
        df['target_item_attributes'] = df['item_id'].apply(lambda x: item_df.loc[x]['item_attributes'])
        # Candidates id
        df['candidate_item_id'] = df.apply(lambda x: [x['item_id']]+x['neg_item_id'], axis=1)

        def shuffle_list(x):
            random.shuffle(x)
            return x

        df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: shuffle_list(x))  # shuffle candidates id

        # Add item attributes
        def candidate_attr(x):
            candidate_item_attributes = []
            for item_id, item_attributes in zip(x, item_df.loc[x]['item_attributes']):
                candidate_item_attributes.append(f'{item_id}: {item_attributes}')
            return candidate_item_attributes

        df['candidate_item_attributes'] = df['candidate_item_id'].apply(lambda x: candidate_attr(x))
        df['candidate_item_attributes'] = df['candidate_item_attributes'].apply(lambda x: '\n'.join(x))
        # Replace empty string with 'None'
        for col in df.columns.to_list():
            df[col] = df[col].apply(lambda x: 'None' if x == '' else x)

    train_df = dfs[0]
    dev_df = dfs[1]
    test_df = dfs[2]
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

def process_data_with_retrieval(dir: str, n_retrieval_items: int = 20):
    """Process the ml-1m raw data using retrieval-based sampling and output the processed data to `dir`.

    Args:
        `dir (str)`: the directory of the dataset. We suppose the raw data is in `dir/raw_data` and the processed data will be output to `dir`.
        `n_retrieval_items (int)`: Number of retrieval items to get for each positive interaction.
    """
    download_data(dir)
    data_df, item_df, user_df = read_data(os.path.join(dir, "raw_data"))
    user_df = process_user_data(user_df)
    logger.info(f'Number of users: {user_df.shape[0]}')
    item_df = process_item_data(item_df)
    logger.info(f'Number of items: {item_df.shape[0]}')
    train_df, dev_df, test_df = process_interaction_data_with_retrieval(data_df, n_retrieval_items)
    logger.info(f'Number of train interactions: {train_df.shape[0]}')
    dfs = append_his_info([train_df, dev_df, test_df], neg=False)  # neg=False since we don't have neg_item_id
    logger.info('Completed append history information to interactions')

    for df in dfs:
        # Format history by listing the historical item attributes
        df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
        # Concat the attributes with item's rating
        df['history'] = df.apply(lambda x: [f'{item_attributes} (rating: {rating})' for item_attributes, rating in zip(x['history'], x['history_rating'])], axis=1)
        # Separate each item attributes by a newline
        df['history'] = df['history'].apply(lambda x: '\n'.join(x))
        # Add user profile for this interaction
        df['user_profile'] = df['user_id'].apply(lambda x: user_df.loc[x]['user_profile'])
        df['target_item_attributes'] = df['item_id'].apply(lambda x: item_df.loc[x]['item_attributes'])
        
        # Shuffle candidate items in-place (candidate_item_id already set in retrieval_sample)
        def shuffle_list(x):
            random.shuffle(x)
            return x

        df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: shuffle_list(x) if len(x) > 0 else [])  # shuffle candidates id

        # Add item attributes
        def candidate_attr(x):
            candidate_item_attributes = []
            for item_id, item_attributes in zip(x, item_df.loc[x]['item_attributes']):
                candidate_item_attributes.append(f'{item_id}: {item_attributes}')
            return candidate_item_attributes

        df['candidate_item_attributes'] = df['candidate_item_id'].apply(lambda x: candidate_attr(x))
        df['candidate_item_attributes'] = df['candidate_item_attributes'].apply(lambda x: '\n'.join(x))
        # Replace empty string with 'None'
        for col in df.columns.to_list():
            df[col] = df[col].apply(lambda x: 'None' if x == '' else x)

    train_df = dfs[0]
    dev_df = dfs[1]
    test_df = dfs[2]
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

if __name__ == '__main__':
    process_data(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ml-1m'))