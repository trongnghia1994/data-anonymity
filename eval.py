import time, datetime
from common import *
from ar_mining import cal_supp_conf
from Apriori import apriori_gen_rules


def eval_results(R_initial, groups, output_file_name, total_time, other_algo=False, k=DESIRED_K):
    print('=========FINAL GROUPS=========')
    if other_algo:
        print('***IGNORED PRINTING GROUPS AND EXPORT DATASET FOR OTHER ALGOS***')
    else:
        pprint_groups(groups, k)
        output_file_name = 'output/' + output_file_name
        export_dataset(groups, output_file_name)

    print('=========ORIGIN RULES=========')
    R_initial.sort(key=lambda rule: rule.hash_value)
    for rule in R_initial:
        pprint_rule(rule)
    print('==============================')
    print('=========RULES MINED ON MODIFIED DATASET=========')
    _, md_rules = apriori_gen_rules(output_file_name)
    md_rules.sort(key=lambda rule: rule.hash_value)    
    for rule in md_rules:
        pprint_rule(rule)
    print('==============================')

    print('=========METRICS=========')
    print('EVALUATED AT {}'.format(datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')))
    print('LENGTH OF R_INITIAL: {}'.format(len(R_initial)))    
    print('RUN TIME: {} seconds'.format(total_time))
    print('Number of groups:', len(groups))
    if other_algo:
        cavg_raw, total_no_tuples, no_unsafe_groups = metrics_cavg_raw(groups, k)    
        print('Number of unsafe groups:', no_unsafe_groups)
        print('Number of tuples:', total_no_tuples)
        print('CAVG:', cavg_raw)
    else:
        print('Number of unsafe groups:', sum(1 for gr in groups if 0 < group_length(gr) < k))
        print('Number of tuples:', sum(group_length(gr) for gr in groups))
        print('CAVG:', metrics_cavg(groups, k))

    no_new_rules, no_loss_rules, no_diff_rules = rules_metrics(R_initial, md_rules)
    print('Number of new rules:', no_new_rules)
    print('Number of loss rules:', no_loss_rules)
    print('Number of diff rules:', no_diff_rules)
