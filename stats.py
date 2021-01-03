from common import RETAINED_DATA_COLUMNS, QUASI_ATTRIBUTES
import pandas

data_file_path = 'dataset/adult-prep.data'
D = pandas.read_csv(data_file_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
dataset_length = D.shape[0]
print('DATASET LENGTH=', dataset_length)
for attr in QUASI_ATTRIBUTES:
    print(attr, len(pandas.unique(D[attr])))
