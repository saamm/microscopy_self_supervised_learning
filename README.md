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
<br>
Currently experimentations is done with Week 1 plate, including Week 2, 3 and perhaps 4 for a rich knowledge base <br>

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
├── 01_exploration.ipynb
├── 02_metadata_analysis.ipynb
├── 03_preprocessing.ipynb
├── 04_baseline_pca.ipynb
├── 05_autoencoder.ipynb
├── 06_ssl_training.ipynb
├── 07_visualization.ipynb
├── 08_metrics.ipynb
│── 09_single_cell_analysis.ipynb   # Maybe
│── 10_biological_interpretation.ipynb
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

Notebook Descriptions

01_exploration.ipynb
Initial data inspection and visualization of raw microscopy images (DAPI, Tubulin, Actin channels). This notebook validates image integrity, resolution consistency, and basic intensity statistics.

02_metadata_analysis.ipynb
Exploration of experimental metadata including compound distributions, concentrations, mechanisms of action (MOA), replicate structure, and plate-level batch effects.

03_preprocessing.ipynb
Prepares microscopy images for representation learning by loading multi-channel images, applying per-channel normalization, resizing to CNN-compatible resolution, and saving processed tensors for downstream modeling.

04_baseline_pca.ipynb
Establishes a classical baseline by flattening images and applying PCA to evaluate whether low-dimensional embeddings capture coarse phenotypic structure and MOA separation.

05_autoencoder.ipynb
Trains a convolutional autoencoder to learn compact latent representations of cellular morphology in an unsupervised manner.

06_ssl_training.ipynb
Implements self-supervised representation learning (e.g., SimCLR-style contrastive learning) to learn biologically meaningful embeddings from microscopy images without using MOA labels during training.

07_visualization.ipynb
Visualizes learned embeddings using UMAP/t-SNE, colored by MOA, compound, and plate to assess biological signal and batch effects.

08_metrics.ipynb
Quantitatively evaluates learned representations via downstream classification of Mechanism of Action (MOA), using proper compound-level cross-validation to avoid information leakage.



