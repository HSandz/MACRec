# MACRec: Multi-Agent Collaboration Framework for Recommendation

A flexible multi-agent system framework for recommendation tasks that supports both **cloud LLMs** (via OpenRouter) and **local LLMs** (via Ollama).

This repository contains the official implementation of our SIGIR 2024 demo paper:
- [Wang, Zhefan, Yuanqing Yu, et al. "MACRec: A Multi-Agent Collaboration Framework for Recommendation". SIGIR 2024.](https://dl.acm.org/doi/abs/10.1145/3626772.3657669)

![framework](./assets/MAC-workflow.png)

## Key Features

- 🤖 **Multi-Agent Collaboration**: Manager, Analyst, Searcher, Interpreter, Reflector, and Retriever agents
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
```

## Project Structure

```
macrec/
├── agents/           # Agent implementations (Manager, Analyst, etc.)
├── llms/            # LLM providers (OpenRouter, Ollama, Gemini, etc.)
├── systems/         # Multi-agent system coordination
├── tasks/           # Task definitions (RP, SR, RR, GEN)
├── tools/           # Agent tools (retrieval, summarization, etc.)
└── utils/           # Utility functions

config/
├── agents/          # Individual agent configurations
├── systems/         # System-level configurations
├── tools/           # Tool configurations
└── api-config.json  # LLM provider settings

data/               # Datasets (MovieLens, Amazon, etc.)
scripts/            # Preprocessing and utility scripts
```

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

| Configuration | Agents | Best For |
|---------------|---------|----------|
| `analyse.json` | Manager + Analyst | Quick testing, simple tasks |
| `retrieve_analyse.json` | Manager + Retriever + Analyst | Tasks needing item retrieval |
| `reflect_analyse_search.json` | Manager + Reflector + Analyst + Searcher | Complex reasoning tasks |
| `full.json` | All 6 agents | Maximum capability |

### Example Workflows

#### 1. Quick Testing (Small Sample)
```bash
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 3 --ollama llama3.2:1b
```

#### 2. Full Evaluation
```bash
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/retrieve_analyse.json --task sr --openrouter google/gemini-2.0-flash-001
```

#### 3. Mixed Provider Setup
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

## Configuration

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
Define which agents to use in `config/systems/collaboration/`:
```json
{
    "agents": ["manager", "analyst"],
    "max_iterations": 3,
    "collaboration_strategy": "sequential"
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
