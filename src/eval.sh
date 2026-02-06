# evaluate lm-probing output
python eval_lm.py --input lm_mistral.jsonl

# evaluate mc-probing output
python eval_mc.py --input mc_mistral.jsonl

# evaluate fr-probing output
python eval_fr.py --input fr_mistral.jsonl