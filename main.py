from common import pick_random_rules, pprint_rule_set
from preprocess import preprocess
from Apriori import apriori_gen_rules
import pickle, subprocess, time

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
    run_oka_algo = ['C:/Python27/python.exe', 'Clustering_based_K_Anon/anonymizer.py', 'a', 'oka', str(k), abs_data_path, abs_output_path]

    commands = [
        # run_m3ar_algo,
        run_modified_algo,
        # run_oka_algo,
    ]
    print(commands)
    for cmd in commands:
        subprocess.run(cmd)

    time.sleep(1)
