## MACRec: a Multi-Agent Collaboration Framework for Recommendation

This repository contains the official implementation of our SIGIR 2024 demo paper:
- [Wang, Zhefan, Yuanqing Yu, et al. "MACRec: A Multi-Agent Collaboration Framework for Recommendation". SIGIR 2024.](https://dl.acm.org/doi/abs/10.1145/3626772.3657669)

The video demo is available at [Video Demo](https://cloud.tsinghua.edu.cn/f/bb41245e81f744fcbd4c/?dl=1).

**A demo of using MACRec**:

https://github.com/wzf2000/MACRec/assets/27494406/0acb4718-5f07-41fd-a06b-d9fb36a7bb1b

![framework](./assets/MAC-workflow.png)

## Key Features

- 🤖 **Multi-Agent Collaboration**: Manager, Analyst, and Reflector agents
- 🧠 **ReWOO Style**: Reasoning Without Observation - 3-phase workflow (Planning → Working → Solving)
- ☁️ **Cloud LLM Support**: Access to 200+ models via OpenRouter plus native OpenAI API integration
- 🏠 **Local LLM Support**: Privacy-focused local inference via Ollama
- 🔧 **Flexible Configuration**: Mix and match different models for different agents
- 🎯 **Multiple Tasks**: Rating Prediction (RP), Sequential Recommendation (SR), Retrieve & Rank (RR), Generation (GEN)
- 📊 **Comprehensive Evaluation**: Built-in metrics and token usage tracking

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n macrec python=3.10.18
conda activate macrec
# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Providers

##### `config/api-config.json` 

```json
{
    "default_provider": "openrouter",
    "providers": {
        "openrouter": {
            "type": "openrouter",
            "api_key": "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "base_url": "https://openrouter.ai/api/v1/chat/completions"
        },
        "openai": {
            "type": "openai",
            "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "base_url": "https://api.openai.com/v1/chat/completions"
        },
        "gemini": {
            "type": "gemini",
            "api_key": "AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models"
        },
        "ollama": {
            "type": "ollama",
            "base_url": "http://localhost:11434"
        }
    }
}
```
---

##### Default Provider

```bash
# Sử dụng provider mặc định (default_provider)
python main.py --main Test \
  --data_file data/ml-100k/test.csv \
  --system rewoo \
  --system_config config/systems/rewoo/basic.json \
  --task sr \
  --samples 100
```

##### Change Default Provider


```json
{
    "default_provider": "openai",  // Thay đổi từ "openrouter" sang "openai"
    "providers": {
        ...
    }
}
```

---

##### Override Provider

```bash
# Sử dụng OpenRouter với model Gemini
python main.py --main Test \
  --data_file data/ml-100k/test.csv \
  --system rewoo \
  --system_config config/systems/rewoo/basic.json \
  --task sr \
  --samples 100 \
  --provider openrouter \
  --model google/gemini-2.0-flash-001

# Sử dụng OpenAI với model GPT-4
python main.py --main Test \
  --data_file data/ml-100k/test.csv \
  --system rewoo \
  --system_config config/systems/rewoo/basic.json \
  --task sr \
  --samples 100 \
  --provider openai \
  --model gpt-4o-mini

# Sử dụng Ollama với model local
python main.py --main Test \
  --data_file data/ml-100k/test.csv \
  --system rewoo \
  --system_config config/systems/rewoo/basic.json \
  --task sr \
  --samples 100 \
  --provider ollama \
  --model llama3.2:1b
```

##### Automatically Skip Providers Without API Key

```json
{
    "providers": {
        "openai": {
            "api_key": ""
        }
    }
}
```


##### Custom Base URL

```json
{
    "openrouter": {
        "base_url": "https://custom-proxy.example.com/api/v1/chat/completions"
    }
}
```
##### Local LLM Setup (Ollama)
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull models: `ollama pull llama3.2:1b`
3. Start server: `ollama serve`

### 3. Prepare Data

```bash
# Windows
scripts\preprocess.bat

# Linux/Mac
bash ./scripts/preprocess.sh
```

#### Generate Embedding-Based Test Files

Create test files with candidates selected from model embeddings (hard negatives):

```bash
# Generate test file with 20 embedding-based candidates
python main.py --main GenerateEmbeddingTest \
    --data_dir data/ml-100k \
    --model_dir models/{model_name}/ml-100k \
    --model {model_name} \
    --n_candidates 20
```

### 4. Run Your First Experiment

```bash
# Test with cloud models (OpenRouter)
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 3 --provider openrouter --model google/gemini-2.0-flash-001

# Test with direct OpenAI models
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 3 --provider openai --model gpt-4o-mini

# Test with local models (Ollama)
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 3 --provider ollama --model llama3.2:1b

# Test ReWOO system (3-phase reasoning workflow)
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 3
```

## Architecture

MACRec features a modern, maintainable architecture built with clean software design principles:

### Core Components

- **🏭 Factory Pattern**: Centralized agent creation with dependency injection
- **🔧 Component Architecture**: Modular orchestrators and coordinators for system workflow management
- **⚙️ Configuration Interface**: Standardized configuration handling with validation
- **🎯 System Orchestration**: ReWOO workflow management

### Available Systems

#### ReWOO System (`--system rewoo`)  
Structured 3-phase reasoning system with optional Manager:
- **Phase 1**: Planner decomposes the task into structured steps
- **Phase 2**: Worker agents execute steps independently  
- **Phase 3**: Solver synthesizes results into final recommendations
- **Architecture**: Uses `ReWOOOrchestrator` for phase-based workflow management
- **Best For**: Complex reasoning tasks requiring systematic analysis


## Project Structure

- `macrec/`: The source folder.
    - `agents/`: All agent classes are defined here.
        - `analyst.py`: The *Analyst* agent class.
        - `base.py`: The base agent class and base tool agent class.
        - `manager.py`: The *Manager* agent class.
        - `reflector.py`: The *Reflector* agent class.
        - `planner.py`: The *Planner* agent for ReWOO-style task decomposition.
        - `solver.py`: The *Solver* agent for ReWOO-style result aggregation.
    - **`components.py`**: Core architectural components (orchestrators, coordinators, state management).
    - `dataset/`: All dataset preprocessing methods.
    - `evaluation/`: The basic evaluation method, including the ranking metrics and the rating metrics.
    - **`factories.py`**: Factory pattern implementation for agent creation and dependency injection.
    - **`config_interface.py`**: Standardized configuration management with validation.
    - `llms/`: The wrapper for LLMs (both API and open source LLMs).
    - `pages/`: The web demo pages are defined here.
    - `rl/`: The datasets and reward function for the RLHF are defined here.
    - `systems/`: The multi-agent system classes are defined here.
        - `base.py`: The base system class with improved architecture.
        - **`rewoo.py`**: The ReWOO system class with component-based architecture and 3-phase workflow.
    - `tasks/`: For external function calls (e.g. main.py). **Note needs to be distinguished from recommended tasks.**
        - `base.py`: The base task class.
        - Task implementations for evaluation, generation, preprocessing, etc.
    - **`components.py`**: Core architectural components (orchestrators, coordinators, state management).
    - `dataset/`: All dataset preprocessing methods.
    - `evaluation/`: The basic evaluation method, including the ranking metrics and the rating metrics.
    - **`factories.py`**: Factory pattern implementation for agent creation and dependency injection.
    - **`config_interface.py`**: Standardized configuration management with validation.
    - `llms/`: The wrapper for LLMs (both API and open source LLMs).
    - `pages/`: The web demo pages are defined here.
    - `rl/`: The datasets and reward function for the RLHF are defined here.
    - `systems/`: The multi-agent system classes are defined here.
        - `base.py`: The base system class with improved architecture.
        - **`rewoo.py`**: The ReWOO system class with component-based architecture and 3-phase workflow.
    - `tasks/`: For external function calls (e.g. main.py). **Note needs to be distinguished from recommended tasks.**
        - `base.py`: The base task class.
        - `calculate.py`: The task for calculating the metrics.
        - `chat.py`: The task for chatting with the `ChatSystem`.
        - **`evaluate.py`**: The task for evaluating the system on the rating prediction or sequence recommendation tasks. The task is inherited from `generation.py`. Auto-adjusts evaluation metrics (HR@10, NDCG@10) for test files with ≥20 candidates.
        - `feedback.py`: The task for selecting the feedback for the *Reflector*. The task is inherited from `generation.py`.
        - `generation.py`: The basic task for generating the answers from a dataset.
        - **`generate_embedding_test.py`**: Generates test CSV files with embedding-based candidates using model embeddings (cosine similarity). Candidates are purely from top-K similar items, excluding user history.
        - `preprocess.py`: The task for preprocessing the dataset.
        - **`pure_generation.py`**: The task for generating the answers from a dataset without any evaluation. The task is inherited from `generation.py`.
        - `reward_update.py`: The task for calculating the reward function for the RLHF.
        - `rlhf.py`: The task for training the *Reflector* with the PPO algorithm.
        - `sample.py`: The task for sampling from the dataset.
        - `test.py`: The task for evaluating the system on few-shot data samples. The task is inherited from `evaluate.py`.
    - `tools/`: Tool implementations for various functionalities.
        - `base.py`: The base tool class.
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
        - `manager_prompt/`: The prompts for the *Manager* with different configurations.
        - `old_system_prompt/`: ***(Deprecated)*** The prompts for other systems' agents.
        - `task_agent_prompt/`: ***(Deprecated)*** The task-specific prompts for agents in other systems.
    - `systems/`: The configuration for each system. Every system has a configuration folder.
    - `tools/`: The configuration for each tool.
    - `training/`: Some configuration for the PPO or other RL algorithms training.
- `ckpts/`: The checkpoint folder for PPO training.
- `data/`: The dataset folder which contains both the raw and preprocessed data.
- `log/`: The log folder.
- `run/`: The evaluation result folder.
- `scripts/`: Some useful scripts.

## Usage


### Available Tasks

| Task | Description | Command Example |
|------|-------------|-----------------|
| **sr** | Sequential Recommendation | `--task sr` |
| **rp** | Rating Prediction | `--task rp` |
| **gen** | Review Generation | `--task gen` |

### System Configurations

#### ReWOO System (3-Phase Reasoning)
| Configuration | Agents | Best For |
|---------------|---------|----------|
| `basic.json` | Planner + Analyst + Solver | Structured reasoning, limited agents |
| `reflector.json` | Planner + Analyst + Reflector + Solver | Recommendations with validation |
| `full.json` | Planner + All Workers + Solver | Complex multi-step reasoning |

### Example Workflows

#### 1. Quick Testing (Small Sample)
```bash
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 3 --provider ollama --model llama3.2:1b
```

#### 2. Full Evaluation
```bash
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --provider openrouter --model google/gemini-2.0-flash-001
```

#### 3. ReWOO System Testing
```bash
# Basic ReWOO with limited agents
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 1 --provider openrouter --model google/gemini-2.0-flash-001

# Full ReWOO with all workers (NOT tested)
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/full.json --task sr --samples 1 --provider openrouter --model google/gemini-2.0-flash-001
```

#### 4. Using Default Model for Provider
If you omit `--model`, the system will use the default model for the specified provider:
```bash
# Uses default model for openrouter (google/gemini-2.0-flash-001)
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 5 --provider openrouter
```

#### 5. Using Agent Config Files (No Override)
If you omit both `--provider` and `--model`, the system will use configurations from individual agent config files:
```bash
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 5
```

## Dataset Support

### Included Datasets
- **MovieLens 100K**: Small dataset for quick testing
- **Amazon Categories**: 24 product categories (Books, Electronics, etc.)
- **Netflix**: Movie ratings dataset
- **Yelp**: Business reviews dataset

### Download Amazon Datasets
```bash
# Download specific Amazon category
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Books --n_neg_items 7

# Available categories: Books, Electronics, Movies_and_TV, Clothing_Shoes_and_Jewelry, 
# Home_and_Kitchen, Sports_and_Outdoors, Beauty, Video_Games, etc.
```

## Advanced Features

### ReWOO System (Reasoning Without Observation)
The ReWOO system implements a structured 3-phase reasoning approach:

1. **Planning Phase**: Planner agent decomposes tasks into sub-problems
2. **Working Phase**: Worker agents execute each step independently  
3. **Solving Phase**: Solver agent aggregates results into final recommendations
4. **Reflection Phase** (Optional): Reflector agent validates workflow execution and result quality

#### Key Benefits
- **Structured Reasoning**: Clear separation of planning, execution, and synthesis
- **Flexible Worker Assignment**: Adapts to available agents (basic vs full configuration)
- **Improved Accuracy**: Multi-step reasoning with intermediate validation
- **Quality Assurance**: Optional Reflector validates completeness and format compliance
- **Scalable**: Can handle complex tasks by breaking them into manageable steps

### Web Demo
```bash
streamlit run web_demo.py
# Visit http://localhost:8501
```

### Token Usage Tracking
All experiments automatically track:
- API calls per agent
- Token usage (input/output)
- Cost estimation
- Model performance metrics
- Detailed logs in `logs/` directory

### Configuration

### System Types

#### ReWOO System (`--system rewoo`)  
Structured 3-phase reasoning system:
- **Phase 1**: Planner decomposes the task
- **Phase 2**: Workers execute steps independently
- **Phase 3**: Solver synthesizes results
- **Architecture**: Component-based with specialized orchestrators for each phase
- **Manager**: Optional (system works without Manager agent)
- **Best For**: Complex reasoning tasks, systematic analysis

### Agent Configuration
Each agent can be configured individually in `config/agents/`:
```json
{
    "provider": "ollama",
    "model": "llama3.2:1b",
    "temperature": 0.7,
    "max_tokens": 1000
}
```

### System Configuration
Define which agents to use and system behavior:

#### ReWOO System  
```json
{
    "agents": ["planner", "analyst", "solver"],
    "max_plan_steps": 5,
    "execution_strategy": "structured"
}
```

## Performance Tips

### Choosing the Right Model

#### Cloud Models (Recommended for Production)
- **High Performance**: `google/gemini-2.0-flash-001`, `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`
- **Balanced**: `meta-llama/llama-3.1-70b-instruct`, `google/gemini-1.5-pro`
- **Budget-Friendly**: `meta-llama/llama-3.1-8b-instruct`, `google/gemini-2.0-flash-001`

#### Local Models (Privacy & Cost-Effective)
- **Testing/Development**: `llama3.2:1b`, `gemma2:2b` (fast, lower quality)
- **Production**: `llama3.1:8b`, `qwen2.5:7b` (good balance)
- **High Quality**: `llama3.1:70b` (requires significant resources)

### Optimization Strategies

#### For Small Models (1B-3B parameters)
- Use simpler prompts
- Reduce max_tokens to 500-1000
- Consider single-agent systems for simple tasks
- Enable response validation and retries

#### For Large Models (70B+ parameters)
- Leverage full multi-agent capabilities
- Use complex reasoning chains
- Enable reflection and self-correction

### Troubleshooting

#### Common Issues
1. **Ollama Connection Error**: Ensure Ollama server is running (`ollama serve`)
2. **Model Not Found**: Pull the model first (`ollama pull <model>`)
3. **Out of Memory**: Try smaller models or reduce batch size
4. **API Rate Limits**: Add delays or use different providers

#### Performance Monitoring
- Check `logs/` directory for detailed execution logs
- Monitor token usage in saved statistics files
- Use `--samples` parameter for testing before full runs

## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{wang2024macrec,
  title={MACRec: A Multi-Agent Collaboration Framework for Recommendation},
  author={Wang, Zhefan and Yu, Yuanqing and Zheng, Wendi and Ma, Weizhi and Zhang, Min},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2760--2764},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
