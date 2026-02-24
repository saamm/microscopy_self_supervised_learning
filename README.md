# microscopy_self_supervised_learning
Self-supervised learning on biological images : Representation Learning for Image-based Drug Profiling in High-Content Microscopy 

## Project Objective:
Can self-supervised representation learning on microscopy images recover biologically meaningful groupings of cellular perturbations without labels?

Optional sub-questions:

Q: Can representations recover Mechanism of Action?
MOA distribution plot
UMAP colored by MOA
MOA classification accuracy

Q: What structure do learned embeddings capture?
Multichannel contrastive learning
Probing tasks
Ablation studies
Discussion of invariances

Q: What cellular morphologies emerge?
Single-cell segmentation
Cell-level embeddings
Phenotype maps
Example cell images per cluster

Comparison of feature types (raw, PCA, SSL)
1. Do SSL embeddings cluster by perturbation?
2. Are they better than PCA / autoencoders?
3. Which phenotypes are confused?

## Data :
Experiments were conducted on a subset of the BBBC021 dataset consisting of three plates from Week 1.

## Structure
```
microscopy-drug-profiling/
│
├── README.md
├── environment.yml
├── requirements.txt
│
├── data/
│   ├── raw/                  # original images + metadata (not pushed to GitHub)
│   ├── processed/            # cleaned metadata, extracted features
│   └── splits/               # train/val/test splits
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_metadata_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_representation_learning.ipynb
│   ├── 05_embedding_analysis.ipynb
│   ├── 06_moa_prediction.ipynb
│   ├── 07_single_cell_analysis.ipynb   # optional advanced
│   └── 08_biological_interpretation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualization.py
│
├── results/
│   ├── figures/
│   │   ├── moa_distribution.png
│   │   ├── concentration_hist.png
│   │   ├── umap_by_moa.png
│   │   ├── umap_by_compound.png
│   │   ├── example_cells.png
│   │   └── channel_comparison.png
│   │
│   ├── tables/
│   │   ├── moa_accuracy.csv
│   │   └── compound_counts.csv
│
├── experiments/
│   ├── config_ssl.yaml
│   ├── config_baseline.yaml
│   └── logs/
│
└── report/
    ├── project_report.pdf
    └── figures/
```



