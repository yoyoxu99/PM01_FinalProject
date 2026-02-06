# lm-probing
python src/run_lm_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --data data/Index4_5_Location.jsonl \
  --output results/lm_mistral.jsonl

# mc-probing without cot
python src/run_mc_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --data data/Index4_5_Location.jsonl \
  --output results/mc_mistral_no_cot.jsonl \
  --try_times 5 \
  --max_new_tokens 32 \
  --top_p 0.9

# mc-probing with cot
python src/run_mc_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --data data/Index4_5_Location.jsonl \
  --output results/mc_mistral_cot.jsonl \
  --try_times 5 \
  --max_new_tokens 32 \
  --top_p 0.9 \
  --cot

# fr-probing without cot
python src/run_fr_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --data data/Index4_5_Location.jsonl \
  --output results/fr_mistral_no_cot.jsonl \
  --max_length 128 \
  --top_p 0.9

# fr-probing with cot
python src/run_fr_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --data data/Index4_5_Location.jsonl \
  --output results/fr_mistral_cot.jsonl \
  --max_length 128 \
  --top_p 0.9 \
  --cot

