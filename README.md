# CaraVaN
This is the pipeline and scoring for the paper:

CaraVaN-Prioritizing functional non-coding variants in Congenital heart disease
Gulrez Chahal, Ngoc Duy Tran, Svenja Pachernegg, Fernando Rossello, Sonika Tyagi*, Mirana Ramialison*

## Description
CaraVaN (Cardiac Variants in Non-coding Genome) is a machine learning-based method to score potentially pathogenic single nucleotide variants in the non-coding genome pertaining to heart disease including congenital heart disease (CHD). We use ensemble approach (Catboost, Light gradient-boosting and Logistic regression) to score genome-wide probability scores for cardiac pathogenicity (where 0 is non-pathogenic and 1 is pathogenic). This model learns froma repertoire of cardiac specific genomic 198 features that have been curated from various genomic resources including cardiac-specific ChIP-sequencing datasets, HiC datasets, transcription factor binding sites.

Languages: Python 3.4, Linux 
