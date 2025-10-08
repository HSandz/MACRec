#! /bin/bash

# Quick test on 100 samples from MovieLens-100k dataset on Sequential Recommendation task
## Evaluate on full dataset using --main Evaluate and without --samples
## Evaluate on Rating Prediction task using --task rp
## Evaluate with other datasets such as Amazon-Beauty by changing --data_file

### config : collaboration = Manager + Analyst
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 100 --openrouter google/gemini-2.0-flash-001
### config : collaboration = Manager + Analyst + Reflector
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/reflect_analyse.json --task sr --samples 100
### config : collaboration = Manager + Analyst + Reflector + Searcher
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/reflect_analyse_search.json --task sr --samples 100
### config : rewoo = Planner + Analyst + Solver
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/basic.json --task sr --samples 100
### config : rewoo = Planner + Analyst + Solver + Reflector
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/reflector.json --task sr --samples 100
### config : rewoo = Planner + Analyst + Solver  + Searcher
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/searcher.json --task sr --samples 100
### config : rewoo = Planner + Analyst + Solver + Reflector + Searcher
python main.py --main Test --data_file data/ml-100k/test.csv --system rewoo --system_config config/systems/rewoo/full.json --task sr --samples 100

### Other params
# --max_his : max history length for sequential recommendation task
# --steps : number of interaction steps between LLM agents
# --topk : number of retrieved documents from the knowledge base
# --samples : number of samples to evaluate, remove this option to evaluate on the full dataset
# --verbose : print the detailed interaction process

# Calculate the metrics directly from the run data file

python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt-gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt-vicu-0.jsonl
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt-vicu-1.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt-gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt-vicu-0.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt-vicu-1.jsonl