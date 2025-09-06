## MACRec: a Multi-Agent Collaboration Framework for Recommendation

This repository contains the official implementation of our SIGIR 2024 demo paper:
- [Wang, Zhefan, Yuanqing Yu, et al. "MACRec: A Multi-Agent Collaboration Framework for Recommendation". SIGIR 2024.](https://dl.acm.org/doi/abs/10.1145/3626772.3657669)

The video demo is available at [Video Demo](https://cloud.tsinghua.edu.cn/f/bb41245e81f744fcbd4c/?dl=1).

**A demo of using MACRec**:

https://github.com/wzf2000/MACRec/assets/27494406/0acb4718-5f07-41fd-a06b-d9fb36a7bb1b

![framework](./assets/MAC-workflow.png)

### File structure

- `macrec/`: The source folder.
    - `agents/`: All agent classes are defined here.
        - `analyst.py`: The *Analyst* agent class.
        - `base.py`: The base agent class and base tool agent class.
        - `interpreter.py`: The *Task Interpreter* agent class.
        - `manager.py`: The *Manager* agent class.
        - `reflector.py`: The *Reflector* agent class.
        - `retriever.py`: The *Retriever* agent class for candidate item retrieval using precomputed embeddings.
        - `searcher.py`: The *Searcher* agent class.
    - `dataset/`: All dataset preprocessing methods.
    - `evaluation/`: The basic evaluation method, including the ranking metrics and the rating metrics.
    - `llms/`: The wrapper for LLMs (both API and open source LLMs).
    - `pages/`: The web demo pages are defined here.
    - `rl/`: The datasets and reward function for the RLHF are defined here.
    - `systems/`: The multi-agent system classes are defined here.
        - `base.py`: The base system class.
        - `collaboration.py`: The collaboration system class. **We recommend using this class for most of the tasks.**
        - `analyse.py`: ***(Deprecated)*** The system with a *Manager* and an *Analyst*. Do not support the `chat` task.
        - `chat.py`: ***(Deprecated)*** The system with a *Manager*, a *Searcher*, and a *Task Interpreter*. Only support the `chat` task.
        - `react.py`: ***(Deprecated)*** The system with a single *Manager*. Do not support the `chat` task.
        - `reflection.py`: ***(Deprecated)*** The system with a *Manager* and a *Reflector*. Do not support the `chat` task.
    - `tasks/`: For external function calls (e.g. main.py). **Note needs to be distinguished from recommended tasks.**
        - `base.py`: The base task class.
        - `calculate.py`: The task for calculating the metrics.
        - `chat.py`: The task for chatting with the `ChatSystem`.
        - **`evaluate.py`**: The task for evaluating the system on the rating prediction or sequence recommendation tasks. The task is inherited from `generation.py`.
        - `feedback.py`: The task for selecting the feedback for the *Reflector*. The task is inherited from `generation.py`.
        - `generation.py`: The basic task for generating the answers from a dataset.
        - `preprocess.py`: The task for preprocessing the dataset.
        - **`pure_generation.py`**: The task for generating the answers from a dataset without any evaluation. The task is inherited from `generation.py`.
        - `reward_update.py`: The task for calculating the reward function for the RLHF.
        - `rlhf.py`: The task for training the *Reflector* with the PPO algorithm.
        - `sample.py`: The task for sampling from the dataset.
        - `test.py`: The task for evaluating the system on few-shot data samples. The task is inherited from `evaluate.py`.
    - `tools/`: Tool implementations for various functionalities.
        - `base.py`: The base tool class.
        - `embedding_retriever.py`: Tool for retrieving top-K items using precomputed embeddings from trained models (e.g., LightGCN).
        - `info_database.py`: Tool for retrieving item attribute information.
        - `interaction.py`: Tool for retrieving user-item interaction history.
        - `summarize.py`: Tool for text summarization.
        - `wikipedia.py`: Tool for Wikipedia information retrieval.
    - `utils/`: Some useful functions are defined here.
- `config/`: The config folder.
    - `api-config.json`: Used for Gemini API configuration. We give an example for the configuration, named `api-config-example.json`.
    - `agents/`: The configuration for each agent.
    - `prompts/`: All the prompts used in the experiments.
        - `agent_prompt/`: The prompts for each agent.
        - `data_prompt/`: The prompts used to prepare the input data for each task.
        - `manager_prompt/`: The prompts for the *Manager* in the `CollaborationSystem` with different configurations.
        - `old_system_prompt/`: ***(Deprecated)*** The prompts for other systems' agents.
        - `task_agent_prompt/`: ***(Deprecated)*** The task-specific prompts for agents in other systems.
    - `systems/`: The configuration for each system. Every system has a configuration folder.
    - `tools/`: The configuration for each tool.
    - `training/`: Some configuration for the PPO or other RL algorithms training.
- `ckpts/`: The checkpoint folder for PPO training.
- `lightgcn/`: The LightGCN implementation for generating user/item embeddings.
    - `config.yaml`: The configuration file for LightGCN training and evaluation.
    - `run.py`: The main file to run LightGCN training and evaluation.
    - `saved/`: The model checkpoint folder for LightGCN.
    - `output/`: The output folder for LightGCN, including the user/item embeddings and ID mapping files.
- `data/`: The dataset folder which contains both the raw and preprocessed data.
- `log/`: The log folder.
- `run/`: The evaluation result folder.
- `scripts/`: Some useful scripts.

### Setup the environment

0. **Create and activate a conda environment with Python 3.10.18:**
    ```shell
    conda create -n macrec python=3.10.18
    conda activate macrec
    ```

1. **Install PyTorch (Note: change the URL setting if using another version of CUDA):**
    ```shell
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
    ```

2. **Install project dependencies:**
    ```shell
    pip install -r requirements.txt
    ```

3. **Download and preprocess datasets:**

   **Quick setup (ml-100k, Amazon):**
   
   **On Windows:**
   ```shell
   scripts\preprocess.bat
   ```
   
   **On Unix/Linux/Mac:**
   ```shell
   bash ./scripts/preprocess.sh
   ```

   **Download specific Amazon datasets:**
   
   You can download and preprocess any Amazon category dataset using the following command:
   ```shell
   python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category {dataset} --n_neg_items 7
   ```

   **Available Amazon Dataset Categories:**

   | Category | Command |
   |----------|---------|
   | Books | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Books --n_neg_items 7` |
   | Electronics | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Electronics --n_neg_items 7` |
   | Movies and TV | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Movies_and_TV" --n_neg_items 7` |
   | CDs and Vinyl | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "CDs_and_Vinyl" --n_neg_items 7` |
   | Clothing, Shoes and Jewelry | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Clothing_Shoes_and_Jewelry" --n_neg_items 7` |
   | Home and Kitchen | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Home_and_Kitchen" --n_neg_items 7` |
   | Kindle Store | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Kindle_Store" --n_neg_items 7` |
   | Sports and Outdoors | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Sports_and_Outdoors" --n_neg_items 7` |
   | Cell Phones and Accessories | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Cell_Phones_and_Accessories" --n_neg_items 7` |
   | Health and Personal Care | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Health_and_Personal_Care" --n_neg_items 7` |
   | Toys and Games | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Toys_and_Games" --n_neg_items 7` |
   | Video Games | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Video_Games" --n_neg_items 7` |
   | Tools and Home Improvement | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Tools_and_Home_Improvement" --n_neg_items 7` |
   | Beauty | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Beauty --n_neg_items 7` |
   | Apps for Android | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Apps_for_Android" --n_neg_items 7` |
   | Office Products | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Office_Products" --n_neg_items 7` |
   | Pet Supplies | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Pet_Supplies" --n_neg_items 7` |
   | Automotive | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Automotive --n_neg_items 7` |
   | Grocery and Gourmet Food | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Grocery_and_Gourmet_Food" --n_neg_items 7` |
   | Patio, Lawn and Garden | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Patio_Lawn_and_Garden" --n_neg_items 7` |
   | Baby | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Baby --n_neg_items 7` |
   | Digital Music | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Digital_Music" --n_neg_items 7` |
   | Musical Instruments | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Musical_Instruments" --n_neg_items 7` |
   | Amazon Instant Video | `python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category "Amazon_Instant_Video" --n_neg_items 7` |

   **Note:** Dataset downloads can take significant time (a few minutes to several hours) depending on the category size and your internet connection. The Books dataset, for example, is approximately 3GB and may take 20-30 minutes to download.

**Note:** We specifically test the code with Python 3.10.18. Other versions may not work as expected. Always activate the conda environment before running any commands:
```shell
conda activate macrec
```

### Training LightGCN Models (just for MovieLens 100k dataset)

To train LightGCN models and generate embeddings for use with the retriever agent:

```shell
python lightgcn/run.py
```

This will:
- Train a LightGCN model on the specified dataset
- You can modify hyperparameters in `lightgcn/config.yaml`
- Save model checkpoints in the `lightgcn/saved/` directory
- Generate and save user/item embeddings in the `lightgcn/output` directory
- Create ID mapping files for the embedding retriever tool in the `lightgcn/output` directory

### API Configuration

This project now uses **OpenRouter.ai with BYOK (Bring Your Own Key)** for all LLM access, including Gemini models.

#### Setup OpenRouter with Your Gemini API Key
1. Go to [OpenRouter.ai](https://openrouter.ai) and create an account
2. Add your Gemini API key to OpenRouter's BYOK settings
3. Get your OpenRouter API key

#### Configure the Project
Create or update `config/api-config.json`:
```json
{
    "provider": "openrouter",
    "api_key": "your-openrouter-api-key-here"
}
```

**Benefits of using OpenRouter with BYOK:**
- Access Gemini models through OpenRouter's unified API
- Use your own Gemini API quota and billing
- Standardized API interface for all models
- Easy switching between different LLM providers

**Note:** Direct Gemini API support has been removed. All Gemini models are now accessed through OpenRouter.ai with your own API keys.

### Run with the command line

Use the following to run specific tasks:
```shell
python main.py -m $task_name --verbose $verbose $extra_args
```

Then `main.py` will run the `${task_name}Task` defined in `macrec/tasks/*.py`.

E.g., to evaluate the sequence recommendation task in MovieLens-100k dataset for the `CollaborationSystem` with *Reflector*, *Analyst*, *Searcher*, and *Retriever* using Gemini (default):
```shell
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/reflect_analyse_search.json --task sr
```

To use different models, specify the `--model` parameter:
```shell
# Use OpenRouter GPT model
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/retrieve_analyse.json --task rr --samples 3 --model openai/gpt-oss-20b:free

# Use Gemini Pro model
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/retrieve_analyse.json --task rr --samples 3 --model gemini-1.5-pro
```

Available model options include:
- **Default**: `gemini` - Gemini 2.0 Flash via OpenRouter (google/gemini-2.0-flash-001)
- `gemini-2.0-flash-001` - Specific Gemini model via OpenRouter
- `gemini-1.5-pro` - Gemini Pro model via OpenRouter
- `google/gemini-2.0-flash-001` - Direct OpenRouter Gemini model name
- `openai/gpt-oss-20b:free` - OpenRouter free GPT model
- `meta-llama/llama-3.1-8b-instruct` - Llama model via OpenRouter
- `meta-llama/llama-3.1-70b-instruct` - Larger Llama model
- `openai/gpt-4o` - GPT-4o via OpenRouter
- `anthropic/claude-3-5-sonnet` - Claude model
- `mistralai/mistral-7b-instruct` - Mistral model
- And many others available on [OpenRouter](https://openrouter.ai/models)

You can refer to the `scripts/` folder for some useful scripts.

### Run with the web demo

Use the following to run the web demo:
```shell
streamlit run demo.py
```

Then open the browser and visit `http://localhost:8501/` to use the web demo.

Please note that the systems utilizing open-source LLMs or other language models may require a significant amount of memory. These systems have been disabled on machines without CUDA support.

### Citation
If you find our work useful, please do not save your star and cite our work:
```
@inproceedings{wang2024macrec,
  title={MACRec: A Multi-Agent Collaboration Framework for Recommendation},
  author={Wang, Zhefan and Yu, Yuanqing and Zheng, Wendi and Ma, Weizhi and Zhang, Min},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2760--2764},
  year={2024}
}
```
