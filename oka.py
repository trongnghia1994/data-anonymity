from common import pick_random_rules, pprint_rule_set, QUASI_ATTRIBUTES, RETAINED_DATA_COLUMNS, metrics_cavg, pprint_rule, metrics_cavg_raw, rules_metrics
from ar_mining import cal_supp_conf
from preprocess import preprocess
from Apriori import apriori_gen_rules
import pickle, subprocess, time, pandas, sys

sys.stdout = open("log/oka_results.log", "a")

if __name__ == '__main__':
    if len(sys.argv) > 3:
        abs_data_path, oka_abs_output_path, initial_rules_path, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    else:
        abs_data_path, oka_abs_output_path, initial_rules_path = '', '', ''

    R_initial = []
    with open(initial_rules_path, 'rb') as f:
        R_initial = pickle.load(f)

    run_oka_algo = ['C:/Python27/python.exe', 'Clustering_based_K_Anon/anonymizer.py', 'a', 'oka', str(k), abs_data_path, oka_abs_output_path]
    subprocess.run(run_oka_algo)

    # Calculate rules support/confidence and Metrics for other algorithms
    print('==ORIGIN RULES==')
    for rule in R_initial:
        pprint_rule(rule)
    # Recalculate support and confidence of rules
    print('==RULES MINED ON MODIFIED DATASET==')
    # modified_R_care = cal_supp_conf(oka_abs_output_path, RETAINED_DATA_COLUMNS, R_initial)
    # for rule in modified_R_care:
    #     pprint_rule(rule)
    out_path, md_rules = apriori_gen_rules(oka_abs_output_path)
    for rule in md_rules:
        pprint_rule(rule)
    # print(rules[:10], len(rules))
    print('=========METRICS=========')        
    dataset = pandas.read_csv(oka_abs_output_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
    GROUPS = dataset.groupby(QUASI_ATTRIBUTES)
    print('Number of groups:', len(GROUPS))
    print('CAVG:', metrics_cavg_raw(GROUPS))
    no_new_rules, no_loss_rules, no_diff_rules = rules_metrics(R_initial, md_rules)
    print('Number of new rules:', no_new_rules)
    print('Number of loss rules:', no_loss_rules)
    print('Number of diff rules:', no_diff_rules)

    sys.stdout.close()