# PM01_FinalProject

📌 Overview
This repository contains the implementation and experimental code for the final course project of Profilierungsmodul Computerlinguistik I – Trustworthy Data-centric AI.
The project focuses on evaluating and analyzing Theory of Mind (ToM) reasoning abilities in Large Language Models (LLMs) using different probing methods.
The main goal of this project is not performance optimization, but to critically examine and improve existing evaluation protocols in the context of trustworthy and data-centric AI.

## 📂 Project Structure

```text
PM01_FinalProject/
├── data/                     # Input datasets (JSONL format)
│   ├── False_Belief_Task.jsonl
│   └── Index4_5_Location.jsonl
│
├── results/                  # Model outputs
│   ├── mc_*.jsonl
│   ├── lm_*.jsonl
│   └── fr_*.jsonl
│
├── src/                      # Source code
│   ├── run_*_model.py          # Run model & generate outputs
│   ├── eval_*.py             # Evaluation scripts
│   ├── lm.py                 # Prompt + Aggregation + Scoring
│   ├── mc.py
│   └── fr.py
│
├── logs/                     # Evaluation logs
│   └── *.txt
│
├── Figures/                  # Figures in thesis
│
├── summary_final.csv         # Final evaluation results
├── fr_avg.csv                # FR average accuracy
└── README.md                 # Project documentation

