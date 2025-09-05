# AttGEI-Net

A deep learning-based system for predicting crop trait performance by integrating genotype data and environmental data to achieve high-precision predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Data Processing](#data-processing)
- [Experimental Design](#experimental-design)
- [Usage Guide](#usage-guide)
- [System Requirements](#system-requirements)

## Project Overview

This project aims to predict crop trait performance for different genotypes under various environmental conditions using deep learning technology. By effectively capturing complex interactions between genotypes and environments, the system delivers accurate crop performance predictions, providing data support for crop breeding and agricultural decision-making.

### Key Features

- Attention mechanism-driven deep learning architecture
- Specialized cross-attention mechanism effectively capturing Genotype-Environment interactions (G×E)
- Comprehensive evaluation system based on multiple experimental designs
- Optimization strategies for different prediction scenarios

## Model Architecture

This project implements an advanced deep learning architecture: AttGEI-Net

### AttGEI-Net (Attention-based Genotype-Environment Interaction Network)

This is a cross-attention model specifically designed for modeling complex genotype-environment interactions. Its core features include:

1. **Multi-head Self-Attention Mechanism** - Uses 8 attention heads to capture richer feature relationships
2. **Deep Feature Extraction** - Three-layer deep encoder networks for genotype and environmental data
3. **Bilinear Interaction Layer** - Explicitly models non-linear relationships between genotype and environment
4. **Hierarchical Feature Fusion** - Combines attention-processed features and direct interaction features

```
Genotype data → Deep Encoder → Self-Attention → Interaction Layer → Feature Fusion → Trait Prediction
Environment data → Deep Encoder ↗          ↗           ↗
                      Bilinear Interaction Layer ─────┘
```

**Model Architecture Details:**
- **Genotype Encoder**: Three-layer fully connected network with LeakyReLU activation, batch normalization, and dropout regularization
- **Environment Encoder**: Same structure as the genotype encoder, specialized for processing environmental features
- **Self-Attention Processing**: Combines genotype and environment features as a sequence, applies multi-head self-attention
- **Interaction Modeling**: Combines attention-processed features and bilinear interaction features
- **Prediction Layer**: Progressively dimensionality-reducing fully connected network, outputs final trait prediction value

## Data Processing

### Data Structure

The project processes three types of data:

1. **Genotype Data** (genodata.txt):
   - Each row represents a sample, columns are genotype markers
   - Matrix of values (-1, 0, or 1)

2. **Environmental Data** (environment/*.xlsx):
   - Environmental indicators for different locations and years
   - Time series data containing various meteorological and soil parameters

3. **Phenotype Data** (characteristic/*.txt):
   - Trait performance of each genotype under specific environments
   - File naming format: "location_year.txt"

### Data Processing Workflow

```
Data loading → Data standardization → Feature extension → Cross-validation splitting
```

- **Standardization**: Robust standardization methods to handle outliers
- **Feature Extension**: Enhanced environmental features, adding statistical and non-linear features
- **Stratified Cross-Validation**: Creating appropriate cross-validation schemes based on experimental design

## Experimental Design

The project employs three complementary experimental designs to comprehensively evaluate model generalization capabilities:

### Experiment 1: Genotype Generalization Testing

- **Objective**: Evaluate model prediction ability for new genotypes
- **Data Split**: All 12 environments, 90% genotypes for training, 10% genotypes for testing
- **Cross-Validation**: 10-fold cross-validation (by genotype)
- **Script**: `main1.py`

### Experiment 2: Environment Generalization Testing

- **Objective**: Evaluate model prediction ability for new environments
- **Data Split**: 5 locations (10 environments) for training, 1 location (2 environments) for testing
- **Cross-Validation**: 6-fold cross-validation (by location)
- **Script**: `main2.py`

### Experiment 3: Comprehensive Generalization Testing

- **Objective**: Evaluate model prediction ability for new genotypes in new environments
- **Data Split**: 5 locations (10 environments) and 90% genotypes for training, 1 location and 10% genotypes for testing
- **Cross-Validation**: 60 combined cross-validation groups (10-fold genotype × 6-fold location)
- **Script**: `main3.py`

## Usage Guide

### Project Structure

```
.
├── basedata/                # Training data folder
│   ├── BPP/                 # Different trait folders
│   │   ├── genodata.txt     # Genotype data
│   │   ├── environment/     # Environmental data folder
│   │   └── characteristic/  # Phenotype data folder
│   └── ...                  # Other traits
├── basedata1/               # Additional training data
├── testdata/                # Testing data folder
├── models.py                # Model definitions
├── utils.py                 # Utility functions
├── trainer.py               # Trainer
├── train_evaluate.py        # Training and evaluation script
├── main1.py                 # Experiment 1 main program
├── main2.py                 # Experiment 2 main program
├── main3.py                 # Experiment 3 main program
└── run_experiment.py        # Experiment execution script
```

### Running Experiments

```bash
# Run Experiment 1 - Genotype generalization
python main1.py

# Run Experiment 2 - Environment generalization
python main2.py

# Run Experiment 3 - Comprehensive generalization
python main3.py

# Run complete experiment workflow
python run_experiment.py --basedata basedata --basedata1 basedata1 --testdata testdata --output results
```

### Interpreting Results

After executing experiments, the following results will be generated:

1. **Training Logs**: Containing training and validation losses for each epoch
2. **Evaluation Metrics**: ARE (Absolute Relative Error), MSE (Mean Squared Error), and Pearson correlation coefficient
3. **Prediction Results**: CSV format prediction result files
4. **Visualizations**: Graphical representations of prediction results and model performance

## System Requirements

### Hardware Requirements

- CPU: 4+ cores (recommended)
- RAM: 8GB+ (16GB recommended)
- GPU: CUDA-compatible GPU (optional but recommended for accelerated training)

### Software Dependencies

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Pandas
- scikit-learn
- Matplotlib (visualization)

### Installing Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib

``` 
