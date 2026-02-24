# microscopy_self_supervised_learning
Self-supervised learning on biological images : Representation Learning for Image-based Drug Profiling in High-Content Microscopy 

## Project Objective:
High-content microscopy enables systematic profiling of drug-induced cellular phenotypes. This project explores whether self-supervised representation learning can recover biological mechanisms of action from multichannel fluorescence images.

## Scientific Questions
Can learned embeddings separate drugs by MOA? <br>
What invariances are captured by contrastive learning across channels? <br>
What morphological phenotypes emerge at single-cell level? <br>

## Data :
BBBC021 <br>
3 channels: DAPI, Tubulin, Actin <br>
X compounds, Y MOAs. <br>

## Methods
Image preprocessing <br>
Contrastive learning (SimCLR / BYOL style) <br>
UMAP visualization <br>
MOA classification <br>
Single-cell clustering <br>

## Results


## Biological Interpretation


## Future Work







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



