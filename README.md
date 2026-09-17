# COVID-19 Strain Classifier Comparison

**CSE 3810 — Computational Genomics**

A machine learning pipeline for classifying SARS-CoV-2 variants (Alpha, Beta, Delta, Gamma, Omicron) from genomic sequence data. Three classifiers are trained on k-mer encoded sequences and compared on classification performance.

## Overview

Viral genome sequences differ in ways that can be captured computationally without full alignment or phylogenetic analysis. This project explores whether simple sequence-composition features (k-mers) are enough to reliably distinguish between COVID-19 variants using standard ML classifiers.

Separating the genome from the rest of the .fasta file required data pasring and output as a .csv file including the genome as well as its label. Parse_strains.py was needed to get the data ready for covid_strain_classifier.py. 

## Data Pipeline

1. Parse FASTA files using Biopython
2. Encode sequences using 4-mers
3. Split data 80/20 into training and testing sets
4. Train models (Multinomial Naive Bayes, Extra Trees, XGBoost)
5. Evaluate using accuracy, precision, recall, and F1-score
6. Test on mystery dataset
7. Test on a single random mystery sequence

## Repository Structure

```
.
├── parse_strains.py     # FASTA parsing + 4-mer encoding
├── covid_strain_classifier.py          # Model training, evaluation, prediction, and final data visualization
└── README.md
```

## Models Compared

| Model | Type |
|---|---|
| Multinomial Naive Bayes | Probabilistic |
| Extra Trees Classifier | Ensemble (bagging) |
| XGBoost | Ensemble (boosting) |

## Getting Started

### Prerequisites

```bash
pip install biopython scikit-learn xgboost numpy pandas
```

### Usage

1. Place your FASTA file(s) in the project directory.
2. Run the parser to generate encoded features:
   ```bash
   python parse_strains.py
   ```
3. Train and evaluate the models:
   ```bash
   python covid_strain_classifier.py
   ```

### Results

<img width="726" height="341" alt="image" src="https://github.com/user-attachments/assets/c81d7722-2630-48da-8318-a3ea3255e75e" />


<img width="527" height="138" alt="image" src="https://github.com/user-attachments/assets/828ceec0-0f91-4cef-843a-451db1751652" />

