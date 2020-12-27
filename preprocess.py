import pandas, numpy as np

from common import DATA_COLUMNS, RETAINED_DATA_COLUMNS
OUTPUT_PATH = 'dataset/adult-prep.data'


def preprocess(input_file_path='dataset/adult.data'):
    df = pandas.read_csv(input_file_path, names=DATA_COLUMNS, index_col=False, skipinitialspace=True)
    df = df[RETAINED_DATA_COLUMNS]

    for col in RETAINED_DATA_COLUMNS:
        df = df.drop(df[df[col] == '?'].index)

    output_file_path = input_file_path.split('/')[0] + '/' + input_file_path.split('/')[1].split('.')[0] + '-prep.data'
    df.to_csv(output_file_path, index=False, header=False)
    return output_file_path


if __name__ == '__main__':
    preprocess()
