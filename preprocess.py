import pandas, numpy as np

from common import DATA_FILE_PATH, DATA_COLUMNS
OUTPUT_PATH = 'dataset/adult-prep.data'

df = pandas.read_csv
df = pandas.read_csv(DATA_FILE_PATH, names=DATA_COLUMNS, index_col=False, skipinitialspace=True)

for col in DATA_COLUMNS:
    print(col)
    df = df.drop(df[df[col] == '?'].index)

df.to_csv(OUTPUT_PATH, index=False, header=False)  
