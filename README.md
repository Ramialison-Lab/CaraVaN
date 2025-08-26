# CaraVaN: Cardiac Variants in Non-coding Genome

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://doi.org/pending)

**CaraVaN-Prioritizing functional non-coding variants in Congenital heart disease**

*Gulrez Chahal, Ngoc Duy Tran, Svenja Pachernegg, Fernando Rossello, Sonika Tyagi*, Mirana Ramialison**

## 🔬 Overview

CaraVaN (Cardiac Variants in Non-coding Genome) is a comprehensive machine learning framework designed to predict the pathogenicity of single nucleotide variants (SNVs) in non-coding genomic regions associated with cardiac diseases, particularly congenital heart disease (CHD). 

### Key Features

- **Ensemble Machine Learning**: Integrates three complementary algorithms (CatBoost, LightGBM, and Logistic Regression) for robust predictions
- **Cardiac-Specific Features**: Leverages 198 curated genomic features from cardiac-specific datasets
- **Genome-Wide Scoring**: Provides probability scores from 0 (non-pathogenic) to 1 (pathogenic) for variants
- **Comprehensive Pipeline**: End-to-end workflow from data preprocessing to model inference

### 🏗️ Architecture

The CaraVaN framework employs an ensemble approach that combines:
- **CatBoost**: Gradient boosting algorithm optimized for categorical features
- **LightGBM**: Fast gradient boosting framework with high efficiency
- **Logistic Regression**: Linear model providing interpretable baseline predictions

## 📋 Requirements

- **Python**: 3.6 or higher
- **Operating System**: Linux/Unix (recommended for PBS job submissions)
- **Dependencies**: See requirements for specific package versions
- **Memory**: Minimum 16GB RAM recommended for training
- **Storage**: ~50GB for full dataset and model weights

## 📁 Repository Structure

```
CaraVaN/
├── README.md                               # This file
├── .gitignore                              # Git ignore patterns
├── model_weights/                          # Pre-trained model files
│   ├── clf_catboost_clinvar.joblib         # CatBoost classifier (ClinVar)
│   ├── clf_catboost_human_encode.joblib    # CatBoost classifier (Human ENCODE)
│   ├── clf_lgbm_clinvar.joblib             # LightGBM classifier (ClinVar)
│   ├── clf_lr_clinvar.joblib               # Logistic Regression (ClinVar)
│   ├── ensemble_human_encode.pkl           # Ensemble model (Human ENCODE)
│   └── [additional model files...]
├── scripts/                                # Python analysis scripts
│   ├── train.py                           # Main training script
│   ├── train_others.py                    # Alternative training pipeline
│   ├── inference.py                       # Inference script
│   ├── mcri_merging_categorical.py        # Feature merging utilities
│   └── [preprocessing scripts...]
├── PBS scripts/                           # High-performance computing scripts
│   ├── training.pbs                       # Training job submission
│   ├── function_inference_chr22_human_encode.pbs  # Chromosome inference
│   └── validation_variants_template.pbs   # Validation pipeline
└── training_data/                         # Training datasets
    ├── 2023_07_28_master_SCREEN_[...].txt # Master feature dataset
    └── Labels new version.txt             # Variant labels
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Ramialison-Lab/CaraVaN.git
cd CaraVaN
```

### 2. Download Training Data

Download the cardiac-specific genomic features dataset:
- **Dataset**: [CaraVaN Datasets](https://bridges.monash.edu/projects/CaraVaN_datasets/118794)
- **Size**: ~5GB compressed
- **Content**: 198 cardiac-specific genomic features from various sources

### 3. Install Dependencies

```bash
pip install -r requirements.txt  # Create requirements.txt with necessary packages
```

## 📊 Training Pipeline

### Phase 1: Data Preparation 🧬

**Objective**: Prepare and quality-control the training dataset

**Key Files**:
- Training data: [`training_data/2023_07_28_master_SCREEN_zebrafirsh_ncer_tfbs_dickelscore_vistaheartenhancers_cvdkp_hic_pchic_atac_fetalheartenhancers.txt`](./training_data/2023_07_28_master_SCREEN_zebrafirsh_ncer_tfbs_dickelscore_vistaheartenhancers_cvdkp_hic_pchic_atac_fetalheartenhancers.txt)
- Labels: [`training_data/Labels new version.txt`](./training_data/Labels%20new%20version.txt)

**Data Sources**:
- ClinVar pathogenic/benign variants
- ENCODE cardiac regulatory elements  
- SCREEN candidate regulatory elements
- Vista heart enhancers
- Hi-C and Promoter Capture Hi-C data
- ATAC-seq cardiac accessibility data
- Transcription factor binding sites

### Phase 2: Feature Engineering 🏗️

**Objective**: Process and merge genomic features using bedtools operations

**Key Scripts**:
- [`scripts/mcri_merging_categorical.py`](./scripts/mcri_merging_categorical.py) - Merge categorical genomic features
- [`scripts/drop_dups_continuous.py`](./scripts/drop_dups_continuous.py) - Remove duplicates from continuous features  
- [`scripts/drop_dups_continuous_dickel.py`](./scripts/drop_dups_continuous_dickel.py) - Process Dickel conservation scores
- [`scripts/validation_variants_merging_categorical.py`](./scripts/validation_variants_merging_categorical.py) - Merge validation set features

**Process**:
1. Intersect variants with genomic annotations using bedtools
2. Handle overlapping features and remove duplicates  
3. Normalize continuous features (conservation scores, accessibility)
4. Encode categorical features (tissue types, regulatory elements)

### Phase 3: Model Training 🤖

**Objective**: Train ensemble models with hyperparameter optimization

**Training Scripts**:
- [`scripts/train.py`](./scripts/train.py) - Primary training pipeline using cardiac disease variants as labels
- [`scripts/train_others.py`](./scripts/train_others.py) - Alternative training using Human ENCODE cREs as labels

**PBS Submission**: 
- [`PBS scripts/training.pbs`](./PBS%20scripts/training.pbs) - High-performance computing job for model training

**Training Process**:
1. **Data Splitting**: Stratified train/validation/test split (70/15/15)
2. **Feature Selection**: Recursive feature elimination and importance ranking
3. **Hyperparameter Tuning**: Grid search with 5-fold cross-validation
4. **Model Training**: Train individual classifiers and ensemble
5. **Model Validation**: Performance assessment on held-out test set

**Output Models**: Saved to [`model_weights/`](./model_weights/) directory
- Individual classifiers (CatBoost, LightGBM, Logistic Regression)
- Ensemble models and voting classifiers
- Feature preprocessing pipelines (PCA, scaling)

### Phase 4: Model Evaluation 📈

**Evaluation Metrics**:
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **PR-AUC**: Area under the precision-recall curve  
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Feature Importance**: SHAP values and permutation importance

## 🔮 Inference Pipeline

### Chromosome-Wide Scoring 🧬

**Objective**: Score variants across entire chromosomes

**Example Script**: [`PBS scripts/function_inference_chr22_human_encode.pbs`](./PBS%20scripts/function_inference_chr22_human_encode.pbs)

**Usage**:
```bash
# Modify chromosome number as needed (e.g., chr22 → chr15)
qsub PBS\ scripts/function_inference_chr22_human_encode.pbs
```

**Key Inference Scripts**:
- [`scripts/inference.py`](./scripts/inference.py) - Main inference pipeline
- [`scripts/inference_blue_green.py`](./scripts/inference_blue_green.py) - Alternative inference method
- [`scripts/inference_others.py`](./scripts/inference_others.py) - Specialized inference pipeline

### Custom Variant Scoring 📊

**Objective**: Score user-provided variant sets

**Pipeline**: [`PBS scripts/validation_variants_template.pbs`](./PBS%20scripts/validation_variants_template.pbs)

**Input Format**: Tab-separated text file with 3 columns:
1. **Chromosome**: Autosomal chromosomes and X chromosome (e.g., chr1, chr2, ..., chrX)  
2. **Start Position**: 0-based start coordinate (inclusive)
3. **End Position**: 0-based end coordinate (exclusive)

**Example**:
```
chr1    12345   12346
chr2    67890   67891  
chrX    11111   11112
```

**Output**: Pathogenicity probability scores (0-1 scale)

## 📁 Model Weights

The [`model_weights/`](./model_weights/) directory contains pre-trained models:

### ClinVar-trained Models
- [`clf_catboost_clinvar.joblib`](./model_weights/clf_catboost_clinvar.joblib) - CatBoost classifier
- [`clf_lgbm_clinvar.joblib`](./model_weights/clf_lgbm_clinvar.joblib) - LightGBM classifier  
- [`clf_lr_clinvar.joblib`](./model_weights/clf_lr_clinvar.joblib) - Logistic Regression
- [`full_pipeline_clinvar.joblib`](./model_weights/full_pipeline_clinvar.joblib) - Complete preprocessing + model pipeline

### Human ENCODE-trained Models  
- [`clf_catboost_human_encode.joblib`](./model_weights/clf_catboost_human_encode.joblib) - CatBoost classifier
- [`ensemble_human_encode.pkl`](./model_weights/ensemble_human_encode.pkl) - Ensemble model
- [`full_pipeline_human_encode.joblib`](./model_weights/full_pipeline_human_encode.joblib) - Complete pipeline

### Specialized Models
- [`clf_full_clinvar_blue_green.joblib`](./model_weights/clf_full_clinvar_blue_green.joblib) - Blue-green deployment model
- [`pca_voting_human_encode.pkl`](./model_weights/pca_voting_human_encode.pkl) - PCA + voting classifier

## 🎯 Use Cases

### 1. Clinical Variant Interpretation
- Prioritize variants of uncertain significance (VUS) in CHD patients
- Complement existing clinical variant interpretation workflows
- Support genetic counseling and diagnosis

### 2. Research Applications  
- Genome-wide association studies (GWAS) follow-up
- Regulatory variant discovery in cardiac development
- Comparative genomics and evolutionary analysis

### 3. Drug Target Discovery
- Identify regulatory variants affecting drug response
- Prioritize therapeutic targets in cardiac diseases

## 📚 Citation

If you use CaraVaN in your research, please cite:

```bibtex
@article{chahal2025caravan,
  title={CaraVaN: Prioritizing functional non-coding variants in Congenital heart disease},
  author={Chahal, Gulrez and Tran, Ngoc Duy and Pachernegg, Svenja and Rossello, Fernando and Tyagi, Sonika and Ramialison, Mirana},
  journal={Journal Name},
  year={2025},
  doi={10.xxxx/xxxxx}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

- **Mirana Ramialison**: mirana.ramialison@mcri.edu.au
- **Sonika Tyagi**: sonika.tyagi@monash.edu
- **Issues**: [GitHub Issues](https://github.com/Ramialison-Lab/CaraVaN/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Murdoch Children's Research Institute (MCRI)
- Monash University
- Australian Research Council (ARC)
- National Health and Medical Research Council (NHMRC)
- All contributors and collaborators

---

**Note for Reviewers**: All scripts, models, and data files referenced in this README are directly linked to their locations within this repository for easy access and reproducibility assessment.
