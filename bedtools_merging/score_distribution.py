# Library declaration
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv("ncER_perc_chr20_coordSorted.txt", sep = '\t', header = None)
df.columns = ['chromosome', 'POS', 'POS_INCR', 'score']

