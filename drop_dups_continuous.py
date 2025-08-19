# Library declaration
import numpy as np
import pandas as pd
import sklearn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chromosome", type=str, required = True)
args = parser.parse_args()

# change the parameters here to update the run 
CHROMOSOME = args.chromosome
VALIDATION_FILE = f"continuous_merging/{CHROMOSOME}_ncer.txt"
OUTPUT_FILE = f"continuous_merging/{CHROMOSOME}_ncer_drop_dups.txt"


print("READ THE NCER FILE")
df_ncer = pd.read_csv(VALIDATION_FILE, sep = '\t', header = None)


# set the column names
print("SETTING COLUMN NAMES")
df_ncer.columns = ['chrom', 'POS', 'POS_INCR', 'chrom_ref', 'POS_REF', 'POS_INCR_REF', 'scores']
df_ncer = df_ncer.drop_duplicates(subset=["chrom", "POS"])
df_ncer = df_ncer.sort_values(by = ['chrom', 'POS', 'POS_INCR'], 
                    ascending = [True, True, True])
print(df_ncer.shape)
print(df_ncer.head())
# df_ncer = df_ncer[['chrom', 'POS', 'POS_INCR', 'scores']]
df_ncer.to_csv(OUTPUT_FILE, sep='\t', header=False, index=False)
