#! /bin/bash

# Evaluate on MovieLens-1m

## Evaluate ReAct
### rating prediction task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system react --system_config config/systems/react/config.json --task rp
### sequential recommendation task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system react --system_config config/systems/react/config.json --task sr --max_his 5

## Evaluate Reflection
### rating prediction task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system reflection --system_config config/systems/reflection/config_api.json --task rp
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system reflection --system_config config/systems/reflection/config_open.json --task rp
### sequential recommendation task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system reflection --system_config config/systems/reflection/config_api.json --task sr --max_his 5
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system reflection --system_config config/systems/reflection/config_open.json --task sr --max_his 5

## Evaluate Manager + Analyst
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task rp --steps 1 --max_his 3
# Evaluate on Amazon-Beauty (1000 samples)

## Evaluate ReAct
### rating prediction task
python main.py --main Evaluate --data_file data/Beauty/test_1000.csv --system react --system_config config/systems/react/config.json --task rp
### sequential recommendation task
python main.py --main Evaluate --data_file data/Beauty/test_1000.csv --system react --system_config config/systems/react/config.json --task sr --max_his 5

## Evaluate Reflection
### rating prediction task
python main.py --main Evaluate --data_file data/Beauty/test_1000.csv --system reflection --system_config config/systems/reflection/config_api.json --task rp
python main.py --main Evaluate --data_file data/Beauty/test_1000.csv --system reflection --system_config config/systems/reflection/config_open.json --task rp
### sequential recommendation task
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --system reflection --system_config config/systems/reflection/config_api.json --task sr --max_his 5
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --system reflection --system_config config/systems/reflection/config_open.json --task sr --max_his 5

## Evaluate Retrieval + Analysis
### retrieval ranking task
#### on MovieLens-100k (100 samples)
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/retrieve_analyse.json --task rr --samples 100
#### on MovieLens full
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/retrieve_analyse.json --task rr
# Calculate the metrics directly from the run data file

python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt-gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt-vicu-0.jsonl
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt-vicu-1.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt-gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt-vicu-0.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt-vicu-1.jsonl
