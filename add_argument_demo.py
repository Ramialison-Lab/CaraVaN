import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--positions', nargs='+', help='<Required> Set flag')
parser.add_argument("-c", "--chromosome", type=str, required = True)

args = parser.parse_args()
positions = args.positions
# positions = [int(ele) for ele in positions]
chromosome = args.chromosome

print(positions)
print(chromosome)