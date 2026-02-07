# PM01_FinalProject

📌 Overview
This repository contains the implementation and experimental code for the final course project of Profilierungsmodul Computerlinguistik I – Trustworthy Data-centric AI.
The project focuses on evaluating and analyzing Theory of Mind (ToM) reasoning abilities in Large Language Models (LLMs) using different probing methods.
The main goal of this project is not performance optimization, but to critically examine and improve existing evaluation protocols in the context of trustworthy and data-centric AI.

📂 Project Structure
PM01_FinalProject/
│
├── data/                  # Input datasets (JSONL format)
│   ├── False_Belief_Task.jsonl # Original datasets
│   └── Index4_5_Location.jsonl
│
├── results/               # Model outputs
│   ├── mc_*.jsonl
│   ├── lm_*.jsonl
│   └── fr_*.jsonl
│
├── src/                   # Source code
│   ├── run_*_model.py     # call model & generate outputs
│   ├── eval_*.py          # evaluation
│   ├── lm.py              # Prompt Templates + Prompt Builder +  Extract Answer + Aggregation + Scoring
│   ├── mc.py
│   └── fr.py
│
├── logs/                  # evaluation performance (.txt)
├── Figures/               # Figures in the thesis
├── summary_final.csv/     # final evaluation performance, from parsing logs
├── fr_avg.csv             # average accuracy of FR probing, from parsing logs
└── README.md              # Project documentation

