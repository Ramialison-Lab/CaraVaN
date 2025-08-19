import pandas as pd 
import numpy as np


# Reading in file
variants_file = "/group/tran3/duytran/ngocduy.tran/Python scripts/bedtools_merging/tested_variants/ensembl_genomic_positions(in).csv"
df = pd.read_csv(variants_file)
df = df[['chromosome_name', 'start_position', 'end_position']]


# Step 1: Add chromosome prefix
df["chrom"] = "chr" + df["chromosome_name"].astype(str)

# Step 2 & 3: Rename columns
df = df.rename(columns={
    "start_position": "POS",
})

df['POS_INCR'] = df['POS'] + 1

# Optional: keep only the final columns
df = df[['chrom', 'POS', 'POS_INCR']]
df = df.drop_duplicates(subset=["chrom", "POS", "POS_INCR"])
df = df.sort_values(by = ['chrom', 'POS', 'POS_INCR'], 
                ascending = [True, True, True])
print(df.shape)

# save file to a 
file_path = "/group/tran3/duytran/ngocduy.tran/Python scripts/bedtools_merging/tested_variants/ensembl_variants_increment.txt"
df.to_csv(file_path, sep='\t', index=False)





# START MERGING WITH CATEGORICAL VARIABLES
"""## Testing for validation file

#### ENCODE merging
"""
print("ENCODE")


import os
import pandas as pd
import numpy as np
import subprocess
import os
import pandas as pd
from datetime import date
import argparse


chromosome = 'ensembl_variants'
logging_file = f"logging/{chromosome}_logging_during_running.txt"

# function to read the human and encode files
# create promoters txt files with: the first initial rows as the original dataframe
# while the most right column is the column produced by the bed file
def read_cres_files_encode(mastersheet,file_lst, folder, subject, start_ind):
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
