# CaraVaN
This is the pipeline and scoring for the paper:

CaraVaN-Prioritizing functional non-coding variants in Congenital heart disease
Gulrez Chahal, Ngoc Duy Tran, Svenja Pachernegg, Fernando Rossello, Sonika Tyagi*, Mirana Ramialison*

## Description
CaraVaN (Cardiac Variants in Non-coding Genome) is a machine learning-based method to score potentially pathogenic single nucleotide variants in the non-coding genome pertaining to heart disease including congenital heart disease (CHD). We use ensemble approach (Catboost, Light gradient-boosting and Logistic regression) to score genome-wide probability scores for cardiac pathogenicity (where 0 is non-pathogenic and 1 is pathogenic). This model learns froma repertoire of cardiac specific genomic 198 features that have been curated from various genomic resources including cardiac-specific ChIP-sequencing datasets, HiC datasets, transcription factor binding sites.

Languages: Python 3.8, Linux 


# Bioinformatics Pipeline Template

## 1. Data Preparation

## 2. Feature Construction

## 3. Data Splitting

## 4. Model Training

## 5. Model Evaluation

## 6. Inference/Prediction

## 7. Visualization & Reporting

## 8. Documentation & Reproducibility

---

## 📊 Bioinformatics Pipeline Overview

<details>
<summary><strong>Click to expand pipeline steps</strong></summary>

### 1. 🧬 Data Preparation
• Collect raw data (<code>VCF</code>, <code>BED</code>, <code>FASTA</code>, etc.)  
• Quality control (QC) and filtering (remove low-quality reads/variants)  
• Data normalization and formatting (convert to required formats)  
• Remove duplicates and handle missing values

### 2. 🏗️ Feature Construction
• Extract relevant features (genomic coordinates, sequence motifs, annotations)  
• Integrate external databases (Ensembl, dbSNP, ClinVar)  
• Encode categorical and continuous features  
• Feature scaling/normalization

### 3. ✂️ Data Splitting
• Split data into <b>training</b>, <b>validation</b>, and <b>test</b> sets  
• Ensure no data leakage (by chromosome, sample, or patient)

### 4. 🤖 Model Training
• Select ML algorithm (<code>CatBoost</code>, <code>LightGBM</code>, <code>Logistic Regression</code>, etc.)  
• Train model on training set  
• Tune hyperparameters (grid search, cross-validation)  
• Evaluate on validation set

### 5. 🧪 Model Evaluation
• Assess performance (accuracy, ROC-AUC, precision/recall, F1-score)  
• Analyze feature importance  
• Perform error analysis

### 6. 🔮 Inference/Prediction
• Apply trained model to new/unseen data  
• Post-process predictions (thresholding, annotation)  
• Output results in desired format

### 7. 📈 Visualization & Reporting
• Visualize results (ROC curves, feature importance plots)  
• Summarize findings in reports or figures

### 8. 📚 Documentation & Reproducibility
• Document all steps, parameters, and software versions  
• Use workflow management tools (Snakemake, Nextflow, CWL) if needed  
• Version control (Git)

</details>


# Things to do:

1. Pipeline
- Finish data pipeline for external
- Finish running chromosome-wide scores
- Instructions

2. Github
- Uploading 
