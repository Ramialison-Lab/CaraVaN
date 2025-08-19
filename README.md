# CaraVaN
This is the pipeline and scoring for the paper:

CaraVaN-Prioritizing functional non-coding variants in Congenital heart disease
Gulrez Chahal, Ngoc Duy Tran, Svenja Pachernegg, Fernando Rossello, Sonika Tyagi*, Mirana Ramialison*

## Description
CaraVaN (Cardiac Variants in Non-coding Genome) is a machine learning-based method to score potentially pathogenic single nucleotide variants in the non-coding genome pertaining to heart disease including congenital heart disease (CHD). We use ensemble approach (Catboost, Light gradient-boosting and Logistic regression) to score genome-wide probability scores for cardiac pathogenicity (where 0 is non-pathogenic and 1 is pathogenic). This model learns froma repertoire of cardiac specific genomic 198 features that have been curated from various genomic resources including cardiac-specific ChIP-sequencing datasets, HiC datasets, transcription factor binding sites.

Languages: Python 3.4, Linux 

## Downloading the cardiac-specific non-coding genome features
The features used in the model can be downloaded from this [link](https://bridges.monash.edu/projects/CaraVaN_datasets/118794).

## Getting the Source Code

To get the source code, please click the "fork" button in the upper-right and then add this repo as an upstream source:

````
$ git clone <your_fork_of_the_repo> ppds
$ cd ppds
$ REPO=https://github.com/Ramialison-Lab/CaraVaN.git
$ git remote add upstream $REPO



