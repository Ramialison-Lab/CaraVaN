import pandas as pd 
import numpy as np

import os
import pandas as pd
import numpy as np
import subprocess
import os
import pandas as pd
from datetime import date
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chromosome", type=str, required = True)
args = parser.parse_args()
chromosome = args.chromosome



variants_file = f"/group/tran3/duytran/ngocduy.tran/Python scripts/bedtools_merging/tested_variants/{chromosome}.txt"
df = pd.read_csv(variants_file, sep ='\t')

# print(df)
# print(df.columns)

# df = df[['Chromosome/scaffold name', 'Start (bp)', 'End (bp)']]
# df

# # Step 2 & 3: Rename columns
# df = df.rename(columns={
#     "Chromosome/scaffold name": "chrom",
#     "Start (bp)": "POS",
#     # 'End (bp)': "POS_INCR"
# })

# df['POS_INCR'] = df['POS'] + 1

# # # Step 1: Add chromosome prefix
# df["chrom"] = "chr" + df["chrom"].astype(str)


# Assume that the file follows the below format

# # Optional: keep only the final columns
df = df[['chrom', 'POS', 'POS_INCR']]
df = df.drop_duplicates(subset=["chrom", "POS", "POS_INCR"])
df = df.sort_values(by = ['chrom', 'POS', 'POS_INCR'], 
                ascending = [True, True, True])
print(df)



# save file to Validation Folder
file_path = f"/group/tran3/duytran/ngocduy.tran/Python scripts/bedtools_merging/Validation/{chromosome}.txt"
df.to_csv(file_path, sep='\t', index=False)



# START MERGING WITH CATEGORICAL VARIABLES
"""## Testing for validation file

#### ENCODE merging
"""
print("ENCODE")





chromosome = args.chromosome
logging_file = f"logging/{chromosome}_logging_during_running.txt"

# function to read the human and encode files
# create promoters txt files with: the first initial rows as the original dataframe
# while the most right column is the column produced by the bed file
def read_cres_files_encode(mastersheet,file_lst, folder, subject, start_ind):
  folder_to_create = f"Validation merging test/{chromosome}"

  try:
    os.makedirs(folder_to_create)
  except OSError:
    pass

  output_odd = f"Validation merging test/{chromosome}/output_odd.txt"
  output_even = f"Validation merging test/{chromosome}/output_even.txt"
  track_ind = start_ind
  for bed_file in file_lst:
    try:

      df = pd.read_csv(os.path.join(folder, bed_file), sep = '\t', header = None)
    except pd.errors.EmptyDataError:
      # handle empty file
      print(f"One missing file: {os.path.join(folder, bed_file)}")
      with open(logging_file, 'a') as fpo:
          fpo.write(f"One missing file: {os.path.join(folder, bed_file)}\n")
      pass
    else:
      name = bed_file[: -4]
    #   print(f"track_ind: {track_ind}")
      print(name)
      # print(bed_file)
      # Run the Bedtools intersect command
      if track_ind == 0:
        output_file = output_even
        command = ['bedtools', 'intersect', '-C', '-a', mastersheet, '-b', os.path.join(folder, bed_file)]
      elif(track_ind %2 ==0):
        output_file = output_even
        command = ['bedtools', 'intersect', '-C', '-a', output_odd, '-b', os.path.join(folder, bed_file)]

      else:
        output_file=  output_odd
        command = ['bedtools', 'intersect', '-C', '-a', output_even, '-b', os.path.join(folder, bed_file)]
      result = subprocess.run(command, capture_output=True, text=True)
      # Save the intersected result to the output file
      with open(output_file, 'w') as f:
          f.write(result.stdout)
      word_count = subprocess.run(["wc", output_file])
      print(word_count.stdout)
      overview = subprocess.run(["sed", "-n", "1p", output_file], capture_output = True, text = True)
      print(overview.stdout)

      # Display the output file name
      print(track_ind)
      print(f"Intersection saved as {name}")
      with open(logging_file, 'a') as fpo:
          fpo.write(name)
          # fpo.write(word_count.stdout)
          fpo.write(overview.stdout)
          fpo.write(str(track_ind))
      track_ind += 1

  return track_ind


def read_bed_files(mastersheet,subject, start_ind):
  folder = f"BED files/{subject}"
  embryo_folder = f"BED files/{subject}/Clean Embryo files"
  files = os.listdir(folder)

  promoters = sorted([ele for ele in files if 'promoter' in ele])
  enhancers = sorted([ele for ele in files if 'enhancer' in ele])
  # for mice, embryo is a redundant column as it is already either the
  # promtoer or enhancer
  if subject == 'Human':
    embryo = sorted([ele for ele in os.listdir(embryo_folder) if 'embryo' in ele])
  ctcf = sorted([ele for ele in files if 'CTCF' in ele])
  print(promoters)
  ind = read_cres_files_encode(mastersheet,promoters, folder, subject, start_ind = start_ind)
  print(f"Starting ind is: {ind}")
  # investigate_levels(enhancers, folder)
  print(enhancers)
  ind = read_cres_files_encode(mastersheet,enhancers, folder, subject, start_ind = ind)
  print(f"Starting ind is: {ind}")

  # investigate_levels(embryo, embryo_folder)
  if subject == 'Human':
    print(embryo)
    ind = read_cres_files_encode(mastersheet,embryo, embryo_folder, subject, start_ind = ind)
    print(f"Starting ind is: {ind}")
  print(ctcf)
  # investigate_levels(ctcf, folder)
  ind = read_cres_files_encode(mastersheet,ctcf, folder, subject, start_ind = ind)
  print(f"Starting ind is: {ind}")

  print(len(promoters), len(enhancers), len(ctcf))
  return ind

# above
validation_file = file_path

bed_starting_ind = read_bed_files(validation_file, subject = "Human", start_ind = 0)
print(f"bed_starting_ind: {bed_starting_ind}")

# bed_starting_ind = 42
bed_starting_ind = read_bed_files(validation_file, subject = 'Mouse', start_ind = bed_starting_ind)
print(f"bed_starting_ind: {bed_starting_ind}") # bed_starting_ind_tfbs = 118
# order: Human - Mouse - TFBS - zebrafish - vista - cvdkp - hic - pchic -atac - genomic (fetal) - ncer - dickel



"""### TFBS"""
print("TFBS")

def read_cres_files(mastersheet,file_lst, folder, start_ind, intersect):
  folder_to_create = f"Validation merging test/{chromosome}"

  try:
    os.makedirs(folder_to_create)
  except OSError:
    pass
    
  output_odd = f"Validation merging test/{chromosome}/output_odd.txt"
  output_even = f"Validation merging test/{chromosome}/output_even.txt"
  track_ind = start_ind
  for bed_file in file_lst:
    try:

      df = pd.read_csv(os.path.join(folder, bed_file), sep = '\t', header = None)
    except pd.errors.EmptyDataError:
      # handle empty file
      print(f"One missing file: {os.path.join(folder, bed_file)}")
      pass
    else:
      name = bed_file[: -4]
      print(name)
      # print(bed_file)
      # Run the Bedtools intersect command
      if track_ind == 0:
        output_file = output_even
        command = ['bedtools', 'intersect', intersect, '-a', mastersheet, '-b', os.path.join(folder, bed_file)]
      elif(track_ind %2 ==0):
        output_file = output_even
        command = ['bedtools', 'intersect', intersect, '-a', output_odd, '-b', os.path.join(folder, bed_file)]

      else:
        output_file=  output_odd
        command = ['bedtools', 'intersect', intersect, '-a', output_even, '-b', os.path.join(folder, bed_file)]
      result = subprocess.run(command, capture_output=True, text=True)
      # Save the intersected result to the output file
      with open(output_file, 'w') as f:
          f.write(result.stdout)

      word_count = subprocess.run(["wc", output_file])
      overview = subprocess.run(["sed", "-n", "1p", output_file], capture_output = True, text = True)
      print(overview.stdout)
      with open(logging_file, 'a') as fpo:
          fpo.write(name)
          fpo.write("\n")
          fpo.write(overview.stdout)
          fpo.write("\n")
          fpo.write(str(track_ind))
          fpo.write("\n")


      # Display the output file name
      print(f"Intersection saved as {name}")
      track_ind += 1

  return track_ind

# example of subfolder: /hg38
def read_general_files(mastersheet,subject, start_ind, sub_folder = "", intersect = '-C'):
  folder = f"BED files/{subject}{sub_folder}"

  if subject in ['CVDKP', 'Hi-C']:
      files = [ele for ele in sorted(os.listdir(folder)) if 'unmapped' not in ele]
  elif subject in ['TFBS']:
      files = [ele for ele in sorted(os.listdir(folder)) if 'bed' in ele]

  else:
      files = [ele for ele in sorted(os.listdir(folder))]

  print(len(files), files)

  ind = read_cres_files(mastersheet,files, folder, start_ind = start_ind, intersect = intersect)
  return ind

# cvdkp and hi-c requires unmapped
bed_starting_ind_tfbs = read_general_files(validation_file, "TFBS", start_ind = bed_starting_ind, intersect = '-C')
# zebrafish is good, vista requires sub_folder, cvdkp requires subfolder, hic is fine, pchic is fine, atac is fine, genomic analysis is fine
# bed_starting_ind_zebra = read_general_files(validation_file, "Zebrafish", start_ind = bed_starting_ind_tfbs, intersect = '-C')
print(f"bed_starting_ind_tfbs: {bed_starting_ind_tfbs}") # bed_starting_ind_tfbs = 118




"""### VISTA"""
print("VISTA")
bed_starting_ind_vista = read_general_files(validation_file, "Vista_human_heart_enhancers", start_ind = bed_starting_ind_tfbs,
                                            intersect = "-C", sub_folder = "/hg38")
print(f"bed_starting_ind_vista: {bed_starting_ind_vista}")




"""### CVDKP"""
print("CVDKP")
bed_starting_ind_cvdkp = read_general_files(validation_file, "CVDKP", start_ind = bed_starting_ind_vista,
                                            intersect = "-C", sub_folder = "/gz lifted bed files")
print(f"bed_starting_ind_cvdkp: {bed_starting_ind_cvdkp}") # bed_starting_ind_cvdkp = 149



"""### ATAC"""
print("ATAC")
bed_starting_ind_atac = read_general_files(validation_file, "ATAC", start_ind = bed_starting_ind_cvdkp, intersect = '-C')
# bed_starting_ind_atac = 187
print(f"bed_starting_ind_atac: {bed_starting_ind_atac}") # bed_starting_ind_atac = 187


"""### HIC AND PCHIC"""
print("HIC AND PCHIC")
bed_starting_ind_hic = read_general_files(validation_file, "Hi-C", start_ind = bed_starting_ind_atac, intersect = '-C')
bed_starting_ind_pchic = read_general_files(validation_file, "PCHiC", start_ind = bed_starting_ind_hic, intersect = '-C')

print(f"bed_starting_ind_pchic: {bed_starting_ind_pchic}") # should be 190

"""### FETAL"""
print("FETAL")
bed_starting_ind_fetal = read_general_files(validation_file, "Genomic analyses clean fetal heart enhancers", start_ind = bed_starting_ind_pchic)
print(f"bed_starting_ind_fetal: {bed_starting_ind_fetal}") # should be 191

# for dickel and ncer
# run bedtools intersection with the chr12_encode_even.txt with ncer and dickel then use paste -d'\t' to append the last column to the dataset




"""### REMOVE DUPLICATES AND KEEP MAXIMUM AND SAVE FILE TO MAX.TXT"""
# newly added to make sure there won't be any duplicates here
print("Removing duplicates and keeping maximum")

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

# remove ncer and dickel
database = encode_features + tfbs_features + vista_feature + cvdkp_feature + atac_feature + hic_feature + pchic_feature + fetal_heart_feature 
print(len(database))

# read and save to output_even.txt
output_even_path = f"Validation merging test/{chromosome}/output_even_snv.txt"
print("Reading in validation_df")
validation_df = pd.read_csv(f"Validation merging test/{chromosome}/output_even.txt", sep = '\t', header = None)
print("Setting column names")
validation_df.columns = ['chrom', 'POS', 'POS_INCR'] + database
# validation_df = validation_df[['chrom', 'POS', 'POS_INCR'] + database]


# sort these by the database features
# Keep row with maximum across all feature columns combined





# print("Computing row-wise max across all database feature columns")
# # compute row-wise max across all database feature columns
# validation_df["feature_max"] = validation_df[database].max(axis=1)

# # group by chrom, POS, POS_INCR and keep only the row with highest feature_max
# validation_df = validation_df.loc[
#     validation_df.groupby(["chrom", "POS", "POS_INCR"])["feature_max"].idxmax()
# ]

# # drop the helper column
# validation_df = validation_df.drop(columns=["feature_max"])




# drop duplicates
validation_df = validation_df.drop_duplicates(subset=["chrom", "POS", "POS_INCR"])






# sort again to keep genome order
validation_df = validation_df.sort_values(by=['chrom','POS','POS_INCR'], 
                                          ascending=[True, True, True])

validation_df.to_csv(output_even_path, sep = '\t', header = None, index = False)
