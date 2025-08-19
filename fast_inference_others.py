# Library declaration
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import os 
import sklearn
from sklearn.decomposition import PCA
import catboost
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, auc
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, auc, confusion_matrix
import lightgbm
from lightgbm import LGBMClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle
import joblib
import argparse

################################################################################
"""
USAGE: this is used as a fast inference using encode (or potentially other columns) 
as the label. Just use this to acquire fast results for chromosome 20 only. For better
optimization, use inference_others.py instead
"""

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chromosome", type=str, required = True)
parser.add_argument("-f", "--features", type=str, required = True)
parser.add_argument('-p','--positions', nargs='+', help='<Required> Set flag')

args = parser.parse_args()

# change the parameters here to update the run 
CHROMOSOME = args.chromosome
label_text = args.features
variants_pos = args.positions
VALIDATION_FILE = f"chromosome_data/{CHROMOSOME}_final_ncer_dickel.txt"
logging_file = f'logging/{CHROMOSOME}_inference_during_running_{label_text}.txt'
vcf_file = f'chromosome_data/vcf_{CHROMOSOME}.txt'
# the pipeline starts from here
print("INFERENCE STARTS NOW")
with open(logging_file, 'w') as fpo:
    fpo.write("INFERENCE STARTS NOW")

# features declaration, with the exact same order as run in merging using bedtools
encode_features = ['heart_left_ventricle_tissue_female_adult_53_years.promoters',
    'heart_right_ventricle_tissue_female_adult_46_years.promoters',
    'heart_right_ventricle_tissue_male_adult_40_years.promoters',
    'heart_left_ventricle_tissue_female_adult_46_years.enhancers',
    'heart_left_ventricle_tissue_male_adult_40_years.enhancers',
    'heart_right_ventricle_tissue_female_adult_46_years.enhancers',
    'heart_right_ventricle_tissue_male_adult_40_years.enhancers',
    'cardiac fibroblast female embryo 94 days and female embryo 98 days',
    'heart left ventricle tissue female embryo 101 days and female embryo 103 days',
    'heart left ventricle tissue female embryo 136 days',
    'heart right ventricle tissue female embryo 101 days and female embryo 103 days',
    'heart tissue embryo 101 days',
    'heart tissue embryo 59 days and female embryo 76 days',
    'heart tissue embryo 80 days',
    'heart tissue embryo 96 days',
    'heart tissue female embryo 103 days',
    'heart tissue female embryo 105 days',
    'heart tissue female embryo 110 days',
    'heart tissue female embryo 116 days and female embryo 98 days',
    'heart tissue female embryo 117 days',
    'heart tissue female embryo 147 days',
    'heart tissue female embryo 91 days',
    'heart tissue male embryo 105 days',
    'heart tissue male embryo 110 days',
    'heart tissue male embryo 120 days',
    'heart tissue male embryo 72 days and male embryo 76 days',
    'heart tissue male embryo 91 days',
    'heart tissue male embryo 96 days',
    'left cardiac atrium tissue female embryo 101 days',
    'heart_left_ventricle_tissue_female_adult_46_years.CTCF-bound cCREs',
    'heart_left_ventricle_tissue_female_adult_51_years.CTCF-bound cCREs',
    'heart_left_ventricle_tissue_female_adult_53_years.CTCF-bound cCREs',
    'heart_left_ventricle_tissue_female_adult_59_years.CTCF-bound cCREs',
    'heart_left_ventricle_tissue_male_adult_40_years.CTCF-bound cCREs',
    'C57BL-6_heart_tissue_embryo_10.5_days clean.promoters',
    'C57BL-6_heart_tissue_male_adult_8_weeks clean.promoters',
    'C57BL-6_heart_tissue_postnatal_0_days clean.promoters',
    'C57BL-6_heart_tissue_embryo_10.5_days clean.enhancers',
    'C57BL-6_heart_tissue_male_adult_8_weeks clean.enhancers',
    'C57BL-6_heart_tissue_postnatal_0_days clean.enhancers',
    'C57BL-6_heart_tissue_male_adult_8_weeks clean.CTCF-bound cCREs',
    'C57BL-6_heart_tissue_postnatal_0_days clean.CTCF-bound cCREs']
mouse_features = encode_features[-8:]
tfbs_features = ['ASCL1_pwmscan_hg38_29347_22951.bed', 
                 'CREB1_pwmscan_hg38_7806_15552.bed', 
                 'EOMES_pwmscan_hg38_45327_5339.bed', 
                 'FOXC1_pwmscan_hg38_14051_14776.bed', 
                 'FOXC2_pwmscan_hg38_30036_45203.bed', 'FOXF1_pwmscan_hg38_28598_34204.bed', 'FOXH1_pwmscan_hg38_43315_3114.bed', 'FOXL1_pwmscan_hg38_43316_13581.bed', 'GATA3_pwmscan_hg38_11738_12391.bed', 'GATA4_pwmscan_hg38_28596_35342.bed', 'GATA5_pwmscan_hg38_9066_18689.bed', 'GATA6_pwmscan_hg38_20055_21663.bed', 'GLI1_pwmscan_hg38_33818_38538.bed', 'GLI2_pwmscan_hg38_7492_29088.bed', 'GLI3_pwmscan_hg38_35563_38491.bed', 'GRHL2_pwmscan_hg38_29347_19788.bed', 'HAND1_pwmscan_hg38_38445_41025.bed', 'HES1_pwmscan_hg38_35815_41265.bed', 'HEY1_pwmscan_hg38_28887_46496.bed', 'HEY2_pwmscan_hg38_43317_44258.bed', 'HIF1A_pwmscan_hg38_15529_13200.bed', 'ISL1_pwmscan_hg38_35815_41742.bed', 'JUN_pwmscan_hg38_38048_1878.bed', 'LBX1_pwmscan_hg38_27664_45938.bed', 'MEF2A_pwmscan_hg38_7806_11341.bed', 'MEF2C_pwmscan_hg38_43317_529.bed', 'MEF2D_pwmscan_hg38_11121_17332.bed', 'MEIS1_pwmscan_hg38_27664_48032.bed', 'MESP1_pwmscan_hg38_38445_42157.bed', 'MIXL1_pwmscan_hg38_27664_42167.bed', 'MSX1_pwmscan_hg38_37655_41051.bed', 'NFAC4_pwmscan_hg38_33560_42902.bed', 'NFATC1_pwmscan_hg38_31217_44715.bed', 'NKX25_pwmscan_hg38_39533_43866.bed', 'NKX3_1_pwmscan_hg38_43163_4491.bed', 'NOTO_pwmscan_hg38_28887_33633.bed', 'OVOL2_pwmscan_hg38_39701_44177.bed', 'PAX8_pwmscan_hg38_44374_46322.bed', 'PITX2_pwmscan_hg38_41734_46461.bed', 'POU4F1_pwmscan_hg38_11435_12160.bed', 'POU4F1_pwmscan_hg38_47659_38526.bed', 'PPARD_pwmscan_hg38_35563_48082.bed', 'PPARG_pwmscan_hg38_27664_10529.bed', 'PRD14_pwmscan_hg38_48118_48439.bed', 'PRDM1_pwmscan_hg38_38048_46851.bed', 'PROX1_pwmscan_hg38_33324_13670.bed', 'RARA_pwmscan_hg38_47659_33241.bed', 'RARB_pwmscan_hg38_48313_3308.bed', 'RBPJ_pwmscan_hg38_10402_12301.bed', 'REST_pwmscan_hg38_11121_3574.bed', 'SHOX2_pwmscan_hg38_44375_4511.bed', 'SIX1_pwmscan_hg38_42746_19293.bed', 'SMAD1_pwmscan_hg38_44445_5787.bed', 'SMAD3_pwmscan_hg38_47659_6575.bed', 'SMAD4_pwmscan_hg38_44288_6824.bed', 'SNAI1_pwmscan_hg38_46684_9294.bed', 'SNAI2_pwmscan_hg38_27665_28556.bed', 'SOX11_pwmscan_hg38_43996_10436.bed', 'SOX17_pwmscan_hg38_9606_11798.bed', 'SOX18_pwmscan_hg38_9606_12295.bed', 'SOX4_pwmscan_hg38_20348_27236.bed', 'SOX9_pwmscan_hg38_11121_9563.bed', 'SRF_pwmscan_hg38_27664_8726.bed', 'TBX1_pwmscan_hg38_38292_542.bed', 'TBX20_pwmscan_hg38_30239_36307.bed', 'TBX2_pwmscan_hg38_30239_37929.bed', 'TBX3_pwmscan_hg38_48118_14186.bed', 'TBX5_pwmscan_hg38_47636_48695.bed', 'TEAD2_pwmscan_hg38_6370_18229.bed', 'TP53_pwmscan_hg38_20055_27716.bed', 'TP53_pwmscan_hg38_4482_5454.bed', 'TWIST1_pwmscan_hg38_38398_47795.bed', 'WT1_pwmscan_hg38_43996_14775.bed', 'YY1_pwmscan_hg38_43315_7423.bed', 'ZBT14_pwmscan_hg38_43996_16382.bed', 'ZIC3_pwmscan_hg38_33324_35656.bed']

# zebra_feature = ['Zebrafish']
vista_feature = ['VISTA_human_heart_enhancers']
cvdkp_feature = ['DFF012AGB.bed', 'DFF017NNG.bed', 'DFF029APW.bed', 'DFF090AWG.bed', 'DFF165PSK.bed', 'DFF201AVG.bed', 'DFF208QZK.bed', 'DFF233YXQ.bed', 'DFF249RGM.bed', 'DFF262NPA.bed', 'DFF307JWU.bed', 'DFF307LLE.bed', 'DFF316NIT.bed', 'DFF367ZAV.bed', 'DFF371HWB.bed', 'DFF444FXW.bed', 'DFF491GMJ.bed', 'DFF512TCX.bed', 'DFF528FIC.bed', 'DFF552AGR.bed', 'DFF559UZT.bed', 'DFF564GOU.bed', 'DFF565QGM.bed', 'DFF621YNV.bed', 'DFF653TBJ.bed', 'DFF729PKJ.bed', 'DFF803PYP.bed', 'DFF842KEC.bed', 'DFF923IAT.bed', 'DFF975AHB.bed']
atac_feature = ['ENCFF024FSS.bed', 'ENCFF055YGM.bed', 'ENCFF072LFQ.bed', 'ENCFF124JSN.bed', 'ENCFF174HPZ.bed', 'ENCFF179JFB.bed', 'ENCFF199VHV.bed', 'ENCFF237JBZ.bed', 'ENCFF239PZJ.bed', 'ENCFF251ADS.bed', 'ENCFF257UAY.bed', 'ENCFF258HEB.bed', 'ENCFF291YRH.bed', 'ENCFF303AKJ.bed', 'ENCFF315UYG.bed', 'ENCFF372VNL.bed', 'ENCFF525ZRG.bed', 'ENCFF529QEX.bed', 'ENCFF556YGF.bed', 'ENCFF560EBG.bed', 'ENCFF567VUC.bed', 'ENCFF643QLU.bed', 'ENCFF718BTP.bed', 'ENCFF720GJU.bed', 'ENCFF731JBR.bed', 'ENCFF783MYY.bed', 'ENCFF792EIF.bed', 'ENCFF811VNG.bed', 'ENCFF820DKN.bed', 'ENCFF830CEC.bed', 'ENCFF890NRK.bed', 'ENCFF898WYV.bed', 'ENCFF901EKY.bed', 'ENCFF953FJE.bed', 'ENCFF968YWL.bed', 'ENCFF984UBD.bed', 'ENCFF988KSK.bed', 'ENCFF991MKU.bed']
hic_feature = ['CM_Merged_DI_TADs', 'CP_Merged_DI_TADs']
pchic_feature = ['PCHiC_interaction_points_sorte_uniq_grch38']
fetal_heart_feature = ["unlifted_fetal_heart_enhancers"]
ncer_feature = ['ncer']
dickel_feature = ['Dickel_et_al_scoreall']

tfbs_features = [feat.replace(".bed", "") for feat in tfbs_features]
cvdkp_feature = [feat.replace(".bed", "") for feat in cvdkp_feature]
atac_feature = [feat.replace(".bed", "") for feat in atac_feature]


database = encode_features + tfbs_features + vista_feature + cvdkp_feature + atac_feature + hic_feature + pchic_feature + fetal_heart_feature + ncer_feature + dickel_feature
print(len(database))
print(database)

#### choosing the cols_as_labels
# construct the mapping dictionary
database_encode = mouse_features + tfbs_features + vista_feature + cvdkp_feature + atac_feature + hic_feature + pchic_feature + fetal_heart_feature + ncer_feature + dickel_feature
database_tfbs = encode_features + vista_feature + cvdkp_feature + atac_feature + hic_feature + pchic_feature + fetal_heart_feature + ncer_feature + dickel_feature

mapping_dict = {
  "human_encode": database_encode,
  "tfbs": database_tfbs
}
implementation = mapping_dict[label_text]













with open(logging_file, 'a') as fpo:
    fpo.write(str(len(database)))
    fpo.write(str(database))
    fpo.write("READING THE CHROMOSOMES DATASET")


# reading the chromosome dataset
print("READING THE CHROMOSOMES DATASET")
validation_df = pd.read_csv(VALIDATION_FILE, sep = '\t', header = None)
# validation_df = pd.read_csv(VALIDATION_FILE, sep = '\t')
print(validation_df)


with open(logging_file, 'a') as fpo:
    fpo.write("READING FILE SUCCESSFULLY")
    # fpo.write(validation_df)


# verify duplicates
# condition = validation_df.duplicated(subset = [0,1,2])
# condition = validation_df.duplicated(subset = ['chrom','POS','POS_INCR'])
# print(f"Duplicates in total: {condition.sum()}")
with open(logging_file, 'a') as fpo:
    fpo.write("SETTING COLUMN NAMES")
# set the column names
print("SETTING COLUMN NAMES")
validation_df.columns = ['chrom', 'POS', 'POS_INCR'] + database
# validation_df = validation_df[['chrom', 'POS', 'POS_INCR'] + database]



###########################################################################
# newly appended: responsible for (encode as label) + (filtering for non coding) 

print("FILTERING OUT FOR NON_CODING VARIANTS")
with open(logging_file, 'a') as fpo:
    fpo.write("FILTERING OUT FOR NON CODING VARS")
print(validation_df.shape)
print(validation_df.head())
# only filtering out for possible non-coding variants: tsa = snv only
# idx_lst is obtained from 
idx_df = pd.read_csv(vcf_file, sep = '\t')
desired_idx = set(list(idx_df['POS']) + variants_pos)
desired_idx = sorted(list(desired_idx))
print(len(desired_idx))
validation_df = validation_df[validation_df['POS'].isin(desired_idx)]
print(validation_df.shape)


print("ENCODE AS LABEL")
with open(logging_file, 'a') as fpo:
    fpo.write("ENCODE AS LABELS")
# encode as label, very important for parsing into the machine learning model
positions = validation_df[['chrom', 'POS', 'POS_INCR']]
validation_df = validation_df.drop(['chrom', 'POS', 'POS_INCR'], axis = 1)
validation_df = validation_df[implementation]
print(f"validation_df.shape: {validation_df.shape}")

################################################################################

with open(logging_file, 'a') as fpo:
    fpo.write("SETTING COLUMN NAMES SUCCESSFULLY")

# load the models 
print("LOADING THE MODEL WEIGHTS")
with open(f'model_weights/pca_voting_{label_text}.pkl', 'rb') as f:
    pca = pickle.load(f)

with open(f'model_weights/ensemble_{label_text}.pkl', 'rb') as f:
    clf_full = pickle.load(f)

full_pipeline = joblib.load(f"model_weights/pipeline_{label_text}.pkl")

with open(logging_file, 'a') as fpo:
    fpo.write("LOADING MODELS SUCCESSFULLY")

# Transformation of PCA

# replace missing values    
print("MISSING VALUES APPENDED")
validation_df['Dickel_et_al_scoreall'] = validation_df['Dickel_et_al_scoreall'].replace('.', 0) # np.nan
validation_df['ncer'] = validation_df['ncer'].replace('.', 0) # np.nan


with open(logging_file, 'a') as fpo:
    fpo.write("MISSING VALUES APPENDED SUCCESSFULLY")

# pipeline transformation
print("FULL PIPELINE TRANSFORMATION")
validation_df = full_pipeline.transform(validation_df)
print("PCA TRANSFORMATION")
validation_df = pca.transform(validation_df)


with open(logging_file, 'a') as fpo:
    fpo.write("TRANSFORMATION SUCCESSFULLY")

print("PREDICTIONS STARTING")
predictions = clf_full.predict(validation_df)
with open(logging_file, 'a') as fpo:
    fpo.write("CLAS PREDICTIONS SUCCESSFULLY \n")
proba_predictions = clf_full.predict_proba(validation_df)
with open(logging_file, 'a') as fpo:
    fpo.write("PROBA PREDICTIONS SUCCESSFULLY\n")
# print(predictions, type(predictions))
print(type(proba_predictions))
print(proba_predictions.shape)

with open(logging_file, 'a') as fpo:
    fpo.write(f"PROBA PREDICTIONS SHAPE: {proba_predictions.shape}\n")

with open(logging_file, 'a') as fpo:
    fpo.write("PREDICTIONS SUCCESSFULLY")
    # fpo.write(proba_predictions.shape)


print("PREDICTIONS SAVING")
with open(logging_file, 'a') as fpo:
    fpo.write("PREDICTIONS SAVING")
np.save(f"predictions/proba_predictions/{CHROMOSOME}_proba_{label_text}.npy", proba_predictions)
# np.save(f"predictions/class_predictions/{CHROMOSOME}_class.npy", predictions)

print("POSITIONS SAVING")
with open(logging_file, 'a') as fpo:
    fpo.write("POSITIONS SAVING")
positions['score'] = proba_predictions[:, 1]
positions.to_csv(f"predictions/proba_predictions/{CHROMOSOME}_scores_{label_text}.txt",
                  index = False, sep = '\t')

print("VISUALIZATION")
# plt.figure()
# plt.hist(proba_predictions, density=True, bins = 100, alpha = 0.5)
# sns.kdeplot(data = proba_predictions)
# plt.savefig(f"plots/{CHROMOSOME}_score_distribution.png", bbox_inches = 'tight')





