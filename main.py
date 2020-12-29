from common import pick_random_rules, pprint_rule_set, QUASI_ATTRIBUTES, RETAINED_DATA_COLUMNS, metrics_cavg, pprint_rule
from ar_mining import cal_supp_conf
from preprocess import preprocess
from Apriori import apriori_gen_rules
import pickle, subprocess, time, pandas

input_ds = 'dataset/adult-min-1000.data'
processed_ds = preprocess(input_ds)
rules_data_file = apriori_gen_rules(processed_ds)
R_initial = pick_random_rules(3, rules_data_file)
picked_rules_file = rules_data_file.split('.')[0] + '-picked.data'
pprint_rule_set(R_initial)
with open(picked_rules_file, 'wb') as f:
    pickle.dump(R_initial, f)

K_SET = [5]
for k in K_SET:
    abs_data_path = 'D:/data_anonymity/' + processed_ds
    run_m3ar_algo = ['python', 'm3ar.py', abs_data_path, picked_rules_file, str(k)]
    run_modified_algo = ['python', 'modified_m3ar.py', abs_data_path, picked_rules_file, str(k)]
    abs_output_path = 'D:/data_anonymity/output/' + 'out_oka_' + 'k_' + str(k) + '_' + processed_ds.split('/')[-1]
    print(abs_data_path, abs_output_path)
    run_oka_algo = ['C:/Python27/python.exe', 'Clustering_based_K_Anon/anonymizer.py', 'a', 'oka', str(k), abs_data_path, abs_output_path]

    commands = [
        # run_m3ar_algo,
        # run_modified_algo,
        run_oka_algo,
    ]
    print(commands)
    for cmd in commands:
        subprocess.run(cmd)

    if run_oka_algo in commands:
        print('====FOR OKA ALGO====')
        output_file_name = 'output/out_oka_k_5_adult-min-1000-prep.data'
        # Calculate rules support/confidence and Metrics for other algorithms
        print('==ORIGIN RULES==')
        for rule in R_initial:
            pprint_rule(rule)
        # Recalculate support and confidence of rules
        print('==RULES MINED ON MODIFIED DATASET==')
        modified_R_care = cal_supp_conf(output_file_name, RETAINED_DATA_COLUMNS, R_initial)
        for rule in modified_R_care:
            pprint_rule(rule)
        print('=========METRICS=========')            
        ds = pandas.read_csv(OUTPUT_DS_PATH, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
        GROUPS = dataset.groupby(QUASI_ATTRIBUTES)
        print('Number of groups:', len(GROUPS))
        print('CAVG:', metrics_cavg(GROUPS))

    time.sleep(1)
