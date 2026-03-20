🧬 Microscopy Self-Supervised Learning

Representation Learning for Image-based Drug Profiling in High-Content Microscopy

🎯 Project Objective

High-content microscopy enables large-scale profiling of drug-induced cellular phenotypes.
This project investigates whether self-supervised learning (SSL) can learn meaningful representations of cellular morphology from multi-channel fluorescence images and recover biological mechanisms of action (MOA) without using labels during training.

A central focus of this work is data efficiency

❓ Scientific Questions

Can self-supervised embeddings separate drugs by MOA under realistic evaluation (LOCO)?

How does dataset size (number of plates) affect representation quality?

What invariances are learned across channels (DAPI, Tubulin, Actin)?

Do embeddings capture biological signal or only dominant morphological variation?

📊 Dataset

BBBC021

3 fluorescence channels:

DAPI (nucleus)

Tubulin (microtubules)

Actin (cytoskeleton)

📦 Data Usage Strategy

Experiments conducted across all 10 plates

Controlled comparison:

Single plate (Week 1) → low-data regime

Multiple plates (Weeks 1–10) → higher-data regime

👉 This enables direct evaluation of:

Impact of dataset scale on SSL representation learning
⚙️ Methods
🔬 Preprocessing

Intensity clipping and normalization

DNA mask-based cropping

Resize to 128×128

Channel-wise scaling to [-1, 1]

🧠 Representation Learning
Self-Supervised Models

SimCLR (ViT backbone)

SimCLR (ResNet18)

with and without ImageNet pretraining

DINO (ViT, teacher–student EMA)

MAE (Masked Autoencoder)

Baselines

PCA (flattened images)

Convolutional Autoencoder

📉 Visualization

PCA → global structure

UMAP → local structure

Colored by:

MOA

Compound

Concentration

📊 Downstream Evaluation

MOA classification (kNN, cosine similarity)

LOCO (Leave-One-Compound-Out) validation

Replicate consistency

Single-cell clustering

🔍 Key Findings (So Far)
🧠 Representation Behavior

SimCLR learns smooth manifolds, not discrete clusters

Weak separation by MOA across all models

Embeddings dominated by global morphology, not mechanism

📉 Data Efficiency Insight (Core Result)

Increasing data (from 1 plate → 10 plates):

Improves stability of embeddings

Does not significantly improve MOA separability

LOCO accuracy remains near random (~5–10%)

👉 Key takeaway:

More data alone does not guarantee biologically meaningful representations in SSL
⚠️ Critical Observation

PCA baseline performs similarly to SSL embeddings

Suggests SSL is not learning additional discriminative biological signal

🧪 Biological Interpretation

Learned features likely capture:

Cell density

Intensity variation

Cell cycle / morphology continuum

Weak alignment with:

Drug mechanism

Functional phenotypes

🚀 Future Work

Improve augmentations (biology-aware, channel-specific)

Multi-scale crops (DINO-style)

Stronger inductive biases (ResNet + pretrained)

Hybrid objectives (SSL + weak supervision)

Batch-effect correction across plates

Investigate why more data ≠ better biology





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



