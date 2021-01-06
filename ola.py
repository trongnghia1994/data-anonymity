import pickle, pandas, sys, time
from common import DATA_FILE_PATH, RETAINED_DATA_COLUMNS, QUASI_ATTRIBUTES

from crowds.kanonymity import ola
from crowds.kanonymity.generalizations import GenRule
from crowds.kanonymity.information_loss import dm_star_loss, prec_loss

from eval import eval_results


def generalize_age(value):
    min = int(value / 10)
    max = min + 10
    return '{}-{}'.format(min, max)


def generalize_workclass(value):
    if value in ['Private']:
        return '*'
    if value in ['Self-emp-not-inc', 'Self-emp-inc']:
        return 'Self-employ'
    if value in ['Federal-gov', 'Local-gov', 'State-gov']:
        return 'gov'
    if value in ['Without-pay', 'Never-worked']:
        return 'not-work'


def generalize_marital_status(value):
    if value in ['Never-married']:
        return '*'
    if value in ['Married-civ-spouse', 'Married-AF-spouse']:
        return 'Married'
    if value in ['Divorced', 'Separated']:
        return 'leave'
    if value in ['Widowed', 'Married-spouse-absent']:
        return 'alone'


def star_generalize(value):    
    return '*'


generalization_rules = {
    'age': GenRule([generalize_age, star_generalize]), # 3-levels generalization
    'sex': GenRule([star_generalize]), # 2-level generalization
    'marital-status': GenRule([generalize_marital_status, star_generalize]),
    'native-country': GenRule([star_generalize]),
    'race': GenRule([star_generalize]),
    'education': GenRule([star_generalize]),    
}          


if __name__ == '__main__':
    if len(sys.argv) > 3:
        abs_data_path, ola_abs_output_path, initial_rules_path, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
        log_to_file = True
    else:
        abs_data_path, initial_rules_path = 'D:/data_anonymity/dataset/adult-min-1000-prep.data', 'adult-min-1000-prep-rules-picked.data'
        k = 10
        ola_abs_output_path = 'D:/data_anonymity/output/out_ola_k_{}_adult-prep.data'.format(k)
        log_to_file = True
    
    if log_to_file:
        sys.stdout = open("log/ola_results_k_" + str(k) + ".log", "w")

    R_initial = []
    with open(initial_rules_path, 'rb') as f:
        R_initial = pickle.load(f)

    start_time = time.time()
    df = pandas.read_csv(DATA_FILE_PATH, names=RETAINED_DATA_COLUMNS,
                    index_col=False, skipinitialspace=True)

    anonymous_df = ola.anonymize(df, k=k, info_loss=prec_loss, generalization_rules=generalization_rules)
    out_file_path = 'output/out_ola_k_{}_adult-prep.data'.format(k)
    # Export dataset
    anonymous_df[0].fillna('*', inplace=True)
    anonymous_df[0].to_csv(ola_abs_output_path, index=False, header=False)

    dataset = pandas.read_csv(ola_abs_output_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
    GROUPS = dataset.groupby(QUASI_ATTRIBUTES)

    total_time = time.time() - start_time

    eval_results(R_initial, GROUPS, ola_abs_output_path, total_time, other_algo=True)

    sys.stdout.close()
