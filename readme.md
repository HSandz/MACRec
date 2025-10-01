## MACRec: a Multi-Agent Collaboration Framework for Recommendation

This repository contains the official implementation of our SIGIR 2024 demo paper:
- [Wang, Zhefan, Yuanqing Yu, et al. "MACRec: A Multi-Agent Collaboration Framework for Recommendation". SIGIR 2024.](https://dl.acm.org/doi/abs/10.1145/3626772.3657669)

The video demo is available at [Video Demo](https://cloud.tsinghua.edu.cn/f/bb41245e81f744fcbd4c/?dl=1).

**A demo of using MACRec**:

https://github.com/wzf2000/MACRec/assets/27494406/0acb4718-5f07-41fd-a06b-d9fb36a7bb1b

![framework](./assets/MAC-workflow.png)

## Key Features

- 🤖 **Multi-Agent Collaboration**: Manager, Analyst, Searcher, Interpreter, Reflector, and Retriever agents
- 🧠 **ReWOO Style**: Reasoning Without Observation - 3-phase workflow (Planning → Working → Solving)
- ☁️ **Cloud LLM Support**: Access to 200+ models via OpenRouter (GPT, Claude, Gemini, Llama, etc.)
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

# Install PyTorch
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup LLM Providers

#### Option A: Cloud Models (OpenRouter)
1. Create account at [OpenRouter.ai](https://openrouter.ai)
2. Get your API key
3. Configure `config/api-config.json`:
```json
{
    "provider": "openrouter",
    "api_key": "your-openrouter-api-key-here"
}
```

#### Option B: Local Models (Ollama)
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

### 4. Run Your First Experiment

```bash
# Test with cloud models (OpenRouter)
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 3 --openrouter google/gemini-2.0-flash-001

# Test with local models (Ollama)
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 3 --ollama llama3.2:1b

# Test ReWOO system (3-phase reasoning workflow)
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 3
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

## Architecture

MACRec features a modern, maintainable architecture built with clean software design principles:

### Core Components

- **🏭 Factory Pattern**: Centralized agent creation with dependency injection
- **🔧 Component Architecture**: Modular orchestrators and coordinators for system workflow management
- **⚙️ Configuration Interface**: Standardized configuration handling with validation
- **🎯 System Orchestration**: Clean separation between collaboration and ReWOO workflows

### Available Systems

#### Collaboration System (`--system collaboration`)
Traditional multi-agent system where agents collaborate dynamically:
- **Manager**: Orchestrates the overall process (required)
- **Worker Agents**: Collaborate on different aspects (analysis, retrieval, interpretation)
- **Architecture**: Uses `AgentCoordinator` and `CollaborationOrchestrator` for workflow management
- **Best For**: General recommendation tasks, complex agent interactions

#### ReWOO System (`--system rewoo`)  
Structured 3-phase reasoning system with optional Manager:
- **Phase 1**: Planner decomposes the task into structured steps
- **Phase 2**: Worker agents execute steps independently  
- **Phase 3**: Solver synthesizes results into final recommendations
- **Architecture**: Uses `ReWOOOrchestrator` for phase-based workflow management
- **Best For**: Complex reasoning tasks requiring systematic analysis

### Deprecated Systems
The following systems have been removed in favor of the improved architecture:
- ~~`analyse`~~ → Use `collaboration` system with `analyse.json` config
- ~~`react`~~ → Use `collaboration` system with single-agent configs  
- ~~`reflection`~~ → Use `collaboration` system with reflection-enabled configs
- ~~`chat`~~ → Use `collaboration` system for chat tasks

## Project Structure

- `macrec/`: The source folder.
    - `agents/`: All agent classes are defined here.
        - `analyst.py`: The *Analyst* agent class.
        - `base.py`: The base agent class and base tool agent class.
        - `interpreter.py`: The *Task Interpreter* agent class.
        - `manager.py`: The *Manager* agent class.
        - `reflector.py`: The *Reflector* agent class.
        - `retriever.py`: The *Retriever* agent class for candidate item retrieval using precomputed embeddings.
        - `searcher.py`: The *Searcher* agent class.
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
        - **`collaboration.py`**: The collaboration system class with factory pattern integration.
        - **`rewoo.py`**: The ReWOO system class with component-based architecture and 3-phase workflow.
    - `tasks/`: For external function calls (e.g. main.py). **Note needs to be distinguished from recommended tasks.**
        - `base.py`: The base agent class and base tool agent class.
        - `interpreter.py`: The *Task Interpreter* agent class.
        - `manager.py`: The *Manager* agent class.
        - `reflector.py`: The *Reflector* agent class.
        - `retriever.py`: The *Retriever* agent class for candidate item retrieval using precomputed embeddings.
        - `searcher.py`: The *Searcher* agent class.
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
        - **`collaboration.py`**: The collaboration system class with factory pattern integration.
        - **`rewoo.py`**: The ReWOO system class with component-based architecture and 3-phase workflow.
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

## Usage

### Command Line Interface

The framework provides flexible CLI options for different LLM providers:

#### Cloud Models (OpenRouter)
```bash
# Use specific OpenRouter model for all agents
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 3 --openrouter google/gemini-2.0-flash-001

# Popular cloud models
--openrouter google/gemini-2.0-flash-001     # Gemini 2.0 Flash
--openrouter openai/gpt-4o                   # GPT-4o
--openrouter anthropic/claude-3-5-sonnet     # Claude 3.5
--openrouter meta-llama/llama-3.1-70b-instruct  # Llama 3.1 70B
```

#### Local Models (Ollama)
```bash
# Use specific Ollama model for all agents
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 3 --ollama llama3.2:1b

# Popular local models
--ollama llama3.2:1b      # Llama 3.2 1B (fast, good for testing)
--ollama llama3.2:3b      # Llama 3.2 3B (balanced)
--ollama llama3.1:8b      # Llama 3.1 8B (high quality)
--ollama gemma2:2b        # Google Gemma 2B
--ollama qwen2.5:7b       # Qwen 2.5 7B
```

#### Mixed Provider Configuration
Use individual agent configurations without CLI overrides:
```bash
# Uses individual agent configs (some agents with Ollama, others with OpenRouter)
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 3
```

### Available Tasks

| Task | Description | Command Example |
|------|-------------|-----------------|
| **sr** | Sequential Recommendation | `--task sr` |
| **rp** | Rating Prediction | `--task rp` |
| **rr** | Retrieve & Rank | `--task rr` |
| **gen** | Review Generation | `--task gen` |

### System Configurations

#### Collaboration System
| Configuration | Agents | Best For |
|---------------|---------|----------|
| `analyse.json` | Manager + Analyst | Quick testing, simple tasks |
| `retrieve_analyse.json` | Manager + Retriever + Analyst | Tasks needing item retrieval |
| `reflect_analyse_search.json` | Manager + Reflector + Analyst + Searcher | Complex reasoning tasks |

#### ReWOO System (3-Phase Reasoning)
| Configuration | Agents | Best For |
|---------------|---------|----------|
| `analyst.json` | Planner + Analyst + Solver | Structured reasoning, limited agents |
| `reflector.json` | Planner + Analyst + Reflector + Solver | Recommendations with validation |
| `analyst_searcher.json` | Planner + Analyst + Searcher + Solver | Genre-aware recommendations |
| `analyst_retriever.json` | Planner + Analyst + Retriever + Solver | Retrieve & rank tasks |
| `full.json` | Planner + All Workers + Solver | Complex multi-step reasoning |

### Example Workflows

#### 1. Quick Testing (Small Sample)
```bash
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 3 --ollama llama3.2:1b
```

#### 2. Full Evaluation
```bash
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/retrieve_analyse.json --task sr --openrouter google/gemini-2.0-flash-001
```

#### 3. ReWOO System Testing
```bash
# Basic ReWOO with limited agents
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 1 --openrouter google/gemini-2.0-flash-001

# Full ReWOO with all workers (NOT tested)
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/full.json --task sr --samples 1 --openrouter google/gemini-2.0-flash-001
```

#### 4. Mixed Provider Setup
Edit agent configs to use different providers, then run:
```bash
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 5
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

#### Usage Examples
```bash
# Use basic ReWOO (Planner + Analyst + Solver)
python main.py --main Test --system rewoo --system_config config/systems/rewoo/analyst.json --task sr --samples 1

# Use ReWOO with Reflection (Planner + Analyst + Solver → Reflector)
python main.py --main Test --system rewoo --system_config config/systems/rewoo/reflector.json --task sr --samples 1
```

### LightGCN Integration
Train embedding models for the Retriever agent:
```bash
cd lightgcn
python run.py
```

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

#### Collaboration System (`--system collaboration`)
Traditional multi-agent system where agents collaborate dynamically:
- **Manager**: Orchestrates the overall process (required)
- **Agents**: Work together on different aspects (analysis, retrieval, interpretation)
- **Architecture**: Uses factory pattern for agent creation and component-based orchestration
- **Best For**: General recommendation tasks, complex interactions

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
    "model_type": "ollama",
    "model_name": "llama3.2:1b",
    "temperature": 0.7,
    "max_tokens": 1000
}
```

### System Configuration
Define which agents to use and system behavior:

#### Collaboration System
```json
{
    "agents": ["manager", "analyst"],
    "max_iterations": 3,
    "collaboration_strategy": "sequential"
}
```

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
2. **Model Not Found**: Pull the model first (`ollama pull model_name`)
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

---

**Need Help?** 
- 📖 Check the `docs/` directory for detailed documentation
- 🐛 Report issues on GitHub
- 💬 Join our community discussions
