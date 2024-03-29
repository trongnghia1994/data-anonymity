from common import pick_random_rules, pprint_rule_set, QUASI_ATTRIBUTES, RETAINED_DATA_COLUMNS, metrics_cavg, pprint_rule, metrics_cavg_raw, rules_metrics
from ar_mining import cal_supp_conf
from preprocess import preprocess
from Apriori import apriori_gen_rules
from eval import eval_results
from oka_py3.anonymizer import run_oka_with_adult_ds
import pickle, subprocess, time, pandas, sys, time

if __name__ == '__main__':
    if len(sys.argv) > 3:
        abs_data_path, oka_abs_output_path, initial_rules_path, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
        log_to_file = True
    else:
        k = 10
        abs_data_path, initial_rules_path = 'dataset/adult-min-100-prep.data', 'adult-min-100-prep-rules-picked.data'
        oka_abs_output_path = 'output/out_oka_k_{}_adult-prep.data'.format(k)
        log_to_file = False
    
    if log_to_file:
        sys.stdout = open("log/oka_results_k_" + str(k) + ".log", "a")

    R_initial = []
    with open(initial_rules_path, 'rb') as f:
        R_initial = pickle.load(f)

    for k in [5]:
        print('K=', k)
        start_time = time.time()

        run_oka_with_adult_ds(abs_data_path, k, oka_abs_output_path, log_to_file)

        dataset = pandas.read_csv(abs_data_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
        GROUPS = dataset.groupby(QUASI_ATTRIBUTES)
        total_time = time.time() - start_time

        eval_results(R_initial, GROUPS, oka_abs_output_path, total_time, other_algo=True, k=k)

    sys.stdout.close()
