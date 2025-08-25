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



# reading process and feature declaration
print("READING PROCESS STARTED")
master = pd.read_csv("mastersheet/2023_07_28_master_SCREEN_zebrafirsh_ncer_tfbs_dickelscore_vistaheartenhancers_cvdkp_hic_pchic_atac_fetalheartenhancers.txt", sep = "\t")
master_copy = master.copy()
# features = master_copy.columns[71: ]
# master_copy = master.copy()
# labels = master_copy[master_copy.columns[-1]]
# # Map values in 'column' to 0 and 1
# master_copy['label'] = master_copy['unlifted_fetal_heart_enhancers'].map(lambda x: 0 if x == 0 else 1)
# features = master_copy[master_copy.columns[29:-2]]
# screen_encode_features = features.columns[:42]
# zebrafish_features = features.columns[42:43] # done
# ncer_features = features.columns[43:44] # done
# tfbs_features = features.columns[44:120]
# dickel_features = features.columns[120:121] # done
# vista_features = features.columns[121:122] # done
# cvdkp_features = features.columns[122:152]
# hic_features = features.columns[152:155]
# # pchic_features = features.columns[154:155]
# atac_features = features.columns[155:193]
# fetal_heart_features = ['unlifted_fetal_heart_enhancers']
# human_encode_samples = screen_encode_features[:-8]
# mouse_encode_samples = screen_encode_features[-8:]
# print(len(human_encode_samples), len(mouse_encode_samples))



# labels impplementation and merge with original dataset
print("LABELLING PROCESS STARTED")
df = pd.read_csv("mastersheet/Labels new version.txt", sep = '\t', header = None, names = ['chrom', 'POS'])
df['chrom'] = 'chr' + df['chrom']
df['POS_INCR'] = df['POS'] + 1
df = df.drop_duplicates(subset = ['chrom', 'POS', 'POS_INCR'])
df['dummy'] = 'dummy'
merged_df = master.merge(df, on=['chrom', 'POS', 'POS_INCR'], how='left')
merged_df['label'] = merged_df['dummy'].notnull().astype(int)
merged_df = merged_df.drop("dummy", axis = 1)
merged_df = merged_df.drop_duplicates()

features = merged_df[merged_df.columns[29:-1]]
label = merged_df['label']
# print(label, label.value_counts())
print(features, features.shape)


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

tfbs_features = ['ASCL1_pwmscan_hg38_29347_22951.bed', 
                 'CREB1_pwmscan_hg38_7806_15552.bed', 
                 'EOMES_pwmscan_hg38_45327_5339.bed', 
                 'FOXC1_pwmscan_hg38_14051_14776.bed', 
                 'FOXC2_pwmscan_hg38_30036_45203.bed', 
                 'FOXF1_pwmscan_hg38_28598_34204.bed', 
                 'FOXH1_pwmscan_hg38_43315_3114.bed',
                 'FOXL1_pwmscan_hg38_43316_13581.bed', 'GATA3_pwmscan_hg38_11738_12391.bed', 'GATA4_pwmscan_hg38_28596_35342.bed', 'GATA5_pwmscan_hg38_9066_18689.bed', 'GATA6_pwmscan_hg38_20055_21663.bed', 'GLI1_pwmscan_hg38_33818_38538.bed', 'GLI2_pwmscan_hg38_7492_29088.bed', 'GLI3_pwmscan_hg38_35563_38491.bed', 'GRHL2_pwmscan_hg38_29347_19788.bed', 'HAND1_pwmscan_hg38_38445_41025.bed', 'HES1_pwmscan_hg38_35815_41265.bed', 'HEY1_pwmscan_hg38_28887_46496.bed', 'HEY2_pwmscan_hg38_43317_44258.bed', 'HIF1A_pwmscan_hg38_15529_13200.bed', 'ISL1_pwmscan_hg38_35815_41742.bed', 'JUN_pwmscan_hg38_38048_1878.bed', 'LBX1_pwmscan_hg38_27664_45938.bed', 'MEF2A_pwmscan_hg38_7806_11341.bed', 'MEF2C_pwmscan_hg38_43317_529.bed', 'MEF2D_pwmscan_hg38_11121_17332.bed', 'MEIS1_pwmscan_hg38_27664_48032.bed', 'MESP1_pwmscan_hg38_38445_42157.bed', 'MIXL1_pwmscan_hg38_27664_42167.bed', 'MSX1_pwmscan_hg38_37655_41051.bed', 'NFAC4_pwmscan_hg38_33560_42902.bed', 'NFATC1_pwmscan_hg38_31217_44715.bed', 'NKX25_pwmscan_hg38_39533_43866.bed', 'NKX3_1_pwmscan_hg38_43163_4491.bed', 'NOTO_pwmscan_hg38_28887_33633.bed', 'OVOL2_pwmscan_hg38_39701_44177.bed', 'PAX8_pwmscan_hg38_44374_46322.bed', 'PITX2_pwmscan_hg38_41734_46461.bed', 'POU4F1_pwmscan_hg38_11435_12160.bed', 'POU4F1_pwmscan_hg38_47659_38526.bed', 'PPARD_pwmscan_hg38_35563_48082.bed', 'PPARG_pwmscan_hg38_27664_10529.bed', 'PRD14_pwmscan_hg38_48118_48439.bed', 'PRDM1_pwmscan_hg38_38048_46851.bed', 'PROX1_pwmscan_hg38_33324_13670.bed', 'RARA_pwmscan_hg38_47659_33241.bed', 'RARB_pwmscan_hg38_48313_3308.bed', 'RBPJ_pwmscan_hg38_10402_12301.bed', 'REST_pwmscan_hg38_11121_3574.bed', 'SHOX2_pwmscan_hg38_44375_4511.bed', 'SIX1_pwmscan_hg38_42746_19293.bed', 'SMAD1_pwmscan_hg38_44445_5787.bed', 'SMAD3_pwmscan_hg38_47659_6575.bed', 'SMAD4_pwmscan_hg38_44288_6824.bed', 'SNAI1_pwmscan_hg38_46684_9294.bed', 'SNAI2_pwmscan_hg38_27665_28556.bed', 'SOX11_pwmscan_hg38_43996_10436.bed', 'SOX17_pwmscan_hg38_9606_11798.bed', 'SOX18_pwmscan_hg38_9606_12295.bed', 'SOX4_pwmscan_hg38_20348_27236.bed', 'SOX9_pwmscan_hg38_11121_9563.bed', 'SRF_pwmscan_hg38_27664_8726.bed', 'TBX1_pwmscan_hg38_38292_542.bed', 'TBX20_pwmscan_hg38_30239_36307.bed', 'TBX2_pwmscan_hg38_30239_37929.bed', 'TBX3_pwmscan_hg38_48118_14186.bed', 'TBX5_pwmscan_hg38_47636_48695.bed', 'TEAD2_pwmscan_hg38_6370_18229.bed', 'TP53_pwmscan_hg38_20055_27716.bed', 'TP53_pwmscan_hg38_4482_5454.bed', 'TWIST1_pwmscan_hg38_38398_47795.bed', 'WT1_pwmscan_hg38_43996_14775.bed', 'YY1_pwmscan_hg38_43315_7423.bed', 'ZBT14_pwmscan_hg38_43996_16382.bed', 'ZIC3_pwmscan_hg38_33324_35656.bed']

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
features = features[database]

# Training and testing partitionting
print("DATA PARTITIONING STARTED")
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = 42, stratify = label)
features_list = x_train.columns
# 19780 for training, 4945 for testing
# train_y ratio: 0.057330637, test_y ratio: 0.0574317492
print("y_train.value_counts()")
print(y_train.value_counts())
print("y_test.value_counts()")
print(y_test.value_counts())
print("x_train.shape and x_test.shape")
print(x_train.shape)
print(x_test.shape)
x_train['Dickel_et_al_scoreall'] = x_train['Dickel_et_al_scoreall'].replace('.', 0) # np.nan
x_test['Dickel_et_al_scoreall'] = x_test['Dickel_et_al_scoreall'].replace('.', 0) # np.nan
x_train['ncer'] = x_train['ncer'].replace('.',0)
x_test['ncer'] = x_test['ncer'].replace('.',0)
x_test_df = x_test.copy()


print("PIPELINE STARTED")
# pipeline for the full model
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

num_attribs = x_train.columns

tune = False
name = 'average'
def train_model(name,  x_train, y_train, tune = False):
  if name == 'lgbm':
    if not tune:
      model = LGBMClassifier(class_weight = None)
      # model = LGBMClassifier(is_unbalance = True)
      model.fit(x_train, y_train)
    else:
      # scale_pos_weight = 16
      model = LGBMClassifier(boosting_type= 'dart', class_weight = 'balanced', n_estimators = 200,
                             learning_rate = 0.2, random_state = 42)
      model.fit(x_train, y_train)

  elif name == 'logistic':
    model = LogisticRegression(class_weight ='balanced', max_iter = 10000)
    model.fit(x_train, y_train)

  elif name == 'svc':
    model = SVC(class_weight = 'balanced',kernel = 'linear', max_iter = 100000, random_state = 42, probability = True)
    model.fit(x_train, y_train)

  elif name == 'catboost':
    model = CatBoostClassifier(\
        n_estimators = 300, max_depth = -1,
    loss_function = "Logloss", scale_pos_weight=16,
    random_seed=42,
    logging_level='Silent')
    model.fit(x_train, y_train)

  elif name == 'average':
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import RandomizedSearchCV

    if not tune:
        # clf1 = LogisticRegression(class_weight ='balanced', max_iter = 10000, random_state = 42)
        # clf2 = LGBMClassifier(boosting_type= 'dart', class_weight = 'balanced', max_depth =  -1, n_estimators = 1000, num_leaves = 2500,
        #                         learning_rate = 0.2, min_child_samples = 30, random_state = 42, min_data_in_leaf = 5)
        # # clf2 = LGBMClassifier(boosting_type= 'dart', class_weight = 'balanced', max_depth =  -1, n_estimators = 1000, learning_rate = 0.2,  min_child_samples = 10, random_state = 42)
        

        # clf3 = CatBoostClassifier(custom_loss="Logloss", scale_pos_weight=16,
        # random_seed=42, logging_level='Silent')

        # model = VotingClassifier(estimators=[('catboost', clf3), ('logistic', clf1), ('lgbm', clf2)],
        #                         voting = 'soft')
        # # model = VotingClassifier(estimators=[('catboost', clf3), ('logistic', clf1), ('lgbm', clf2)], weights = [0.1, 0.75, 0.15],
        # #                 voting = 'soft')
        # model.fit(x_train, y_train)
        # return model
    

        clf1 = LogisticRegression(class_weight ='balanced', max_iter = 10000, random_state = 42)
        clf2 = LGBMClassifier(boosting_type= 'dart', class_weight = 'balanced', max_depth =  -1, n_estimators = 1000, learning_rate = 0.2,  min_child_samples = 10, random_state = 42)

        clf3 = CatBoostClassifier(custom_loss="Logloss", scale_pos_weight=17,
          random_seed=42, logging_level='Silent')

        model = VotingClassifier(estimators=[('catboost', clf3), ('logistic', clf1), ('lgbm', clf2)], weights = [0.1, 0.75, 0.15],
                                voting = 'soft')
        model.fit(x_train, y_train)
        return model

    else: 
        clf1 = LogisticRegression()
        clf2 = LGBMClassifier()

        clf3 = CatBoostClassifier()
        model = VotingClassifier(estimators=[('catboost', clf3), ('logistic', clf1), ('lgbm', clf2)],
                        voting = 'soft')
        params_grid = {
          'catboost__scale_pos_weight': [5, 15, 25], 
          'catboost__learning_rate': [None, 0.001, 0.01, 0.1, 0.2], 
          'catboost__depth': [None, 3,5,10,15], 
          'lgbm__n_estimators': [200, 500, 800, 1000],
          'lgbm__num_leaves': [200, 500, 1000, 2500],
          'lgbm__learning_rate': [0.01, 0.1, 0.2, 0.5],
          'lgbm__min_child_samples': [5, 10, 20, 30], 
          'lgbm__min_data_in_leaf': [2, 5, 10, 20], 
          'logistic__max_iter': [1000, 10000]
        }

        grid = RandomizedSearchCV(estimator = model, 
                                  param_distributions= params_grid, 
                                  scoring='roc_auc', 
                                  cv = 5)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        best_params = grid.best_params_


        

  return model


full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        # ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs)
    ], remainder = 'drop')

x_train_full = full_pipeline.fit_transform(x_train)
x_test_full = full_pipeline.transform(x_test)
pca = PCA(random_state = 42, n_components = 0.95)

print("PCA TRANSFORMATION STARTED")
x_train_full = pca.fit_transform(x_train_full)
x_test_full = pca.transform(x_test_full)

print("TRAINING PROCESS STARTED")
clf_full = train_model(name, x_train_full, y_train, tune = tune)

print("SAVING WEIGHTS")
# saving weights
import pickle
with open("model_weights/ensemble_weights_on_lr.pkl", 'wb') as f:
  pickle.dump(clf_full, f)
with open("model_weights/pca_voting_weights_on_lr.pkl", 'wb') as fp:
  pickle.dump(pca, fp)
joblib.dump(full_pipeline, 'model_weights/pipeline_weights_on_lr.pkl')

print("VISUALIZATION")
# Visualization
# This function is initiated to check whether roc_auc is the same as auc()
def get_scores_proba(model_full, x_train_full,  x_test_full,  y_train, y_test):
    # Predict the test data
    y_pred_full = model_full.predict_proba(x_test_full)[:, 1]
    # print(y_pred_full)
    fpr_full, tpr_full, _ = roc_curve(y_test, y_pred_full)
    # calculate the area under the ROC curve
    AUROC_full = auc(fpr_full, tpr_full)
    # y_pred_full = model_full.predict(x_test_full)
    # Get the scores
    # accuracy = accuracy_score(y_test, y_pred_full)
    # precision = precision_score(y_test, y_pred_full)
    # f1 = f1_score(y_test, y_pred_full)
    # roc_auc = roc_auc_score(y_test, y_pred_full)

    #     # Predict the test datadef
    # y_pred_ncer = model_ncer.predict_proba(x_test_ncer)[:, 1]
    # # print(y_pred_ncer)
    # fpr_ncer, tpr_ncer, _ = roc_curve(y_test, y_pred_ncer)
    # # calculate the area under the ROC curve
    # AUROC_ncer = auc(fpr_ncer, tpr_ncer)

    # y_pred_others = model_others.predict_proba(x_test_others)[:, 1]
    # # print(y_pred_ncer)
    # fpr_others, tpr_others, _ = roc_curve(y_test, y_pred_others)
    # # calculate the area under the ROC curve
    # AUROC_others = auc(fpr_others, tpr_others)


    cm = confusion_matrix(y_test, model_full.predict(x_test_full))
    print(cm)
    cmap = sns.diverging_palette(600, 10, as_cmap=True)

    plt.figure(figsize=(10,8), dpi = 600)
    #Plot the confusion matrix.
    sns.heatmap(cm,
                annot=True,
                fmt='g', cmap = cmap)
    plt.ylabel('Actual',fontsize=13)
    plt.xlabel('Prediction',fontsize=13)
    plt.title('Ensembling confusion matrix analysis',fontsize=17)
    plt.savefig("plots/voting_confusion.png", bbox_inches = 'tight', dpi = 600)

    # Plot the ROC curve
    plt.figure(figsize=(10,8), dpi = 600)
    fpr_full, tpr_full, thresholds_full = roc_curve(y_test, y_pred_full)
    # fpr_ncer, tpr_ncer, thresholds_ncer = roc_curve(y_test, y_pred_ncer)
    # fpr_others, tpr_others, thresholds_others = roc_curve(y_test, y_pred_others)

    plt.plot(fpr_full, tpr_full, label = f'(AUC_full: {AUROC_full:.3f})')
    # plt.plot(fpr_others, tpr_others, label = f'(AUC_others: {AUROC_others:.3f})')
    # plt.plot(fpr_ncer, tpr_ncer, label = f'(AUC_ncer: {AUROC_ncer:.3f})')

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig("plots/voting_roc.png", bbox_inches = 'tight', dpi = 600)

# get_scores_proba(clf_full,  x_train_ncer,  x_test_ncer,y_train, y_test)
get_scores_proba(clf_full,  x_train_full,  x_test_full,y_train, y_test)

# print(pd.DataFrame(clf.predict(x_test)).value_counts())
print("Prediction probability duplicates sorted in desceding order")
print(pd.DataFrame(clf_full.predict_proba(x_test_full)).value_counts())

# for row in pd.DataFrame(clf.predict_proba(x_test)).value_counts():
#   print(row)
dups = [ele for ele in pd.DataFrame(clf_full.predict_proba(x_test_full)).value_counts() if ele > 2]
print(f'Number of duplicates in probability prediction: {np.sum(dups)} / {x_test_full.shape[0]} instances')

