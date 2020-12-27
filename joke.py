import pickle, pandas
from common import DATA_FILE_PATH, RETAINED_DATA_COLUMNS

# Read list of rules from binary pickled file
# with open('initial_rules.data', 'rb') as f:
#     data = pickle.load(f)
#     print(data)

df = pandas.read_csv(DATA_FILE_PATH, names=RETAINED_DATA_COLUMNS,
                    index_col=False, skipinitialspace=True)
a = df['education'].unique()
for i in sorted(a):
    print('{};*'.format(i))
