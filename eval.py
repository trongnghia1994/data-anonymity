import time, datetime
from common import *
from ar_mining import cal_supp_conf
from Apriori import apriori_gen_rules


def eval_results(R_initial, groups, output_file_name, start_time, other_algo=False):
    total_time = time.time() - start_time
    print('================================================================================')

    print('LENGTH OF R_INITIAL: {}'.format(len(R_initial)))
    print('EVALUATED AT {}'.format(datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')))
    print('RUN TIME: {} seconds'.format(total_time))

    print('=========FINAL GROUPS=========')
    if other_algo:
        print('***IGNORED PRINTING GROUPS AND EXPORT DATASET FOR OTHER ALGOS***')
    else:
        pprint_groups(groups)
        output_file_name = 'output/' + output_file_name
        export_dataset(groups, output_file_name)

    print('=========ORIGIN RULES=========')
    R_initial.sort(key=lambda rule: rule.hash_value)
    for rule in R_initial:
        pprint_rule(rule)
    # Recalculate support and confidence of rules    
    modified_R_initial = cal_supp_conf(output_file_name, RETAINED_DATA_COLUMNS, R_initial)
    no_existing_rules = 0
    for rule in modified_R_initial:
        if rule.support >= MIN_SUP and rule.confidence >= MIN_CONF:
            no_existing_rules += 1

    print('=========RULES MINED ON MODIFIED DATASET=========')
    _, md_rules = apriori_gen_rules(output_file_name)
    md_rules.sort(key=lambda rule: rule.hash_value)    
    for rule in md_rules:
        pprint_rule(rule)

    print('=========METRICS=========')
    print('Number of groups:', len(groups))
    if other_algo:
        cavg_raw, total_no_tuples = metrics_cavg_raw(groups)    
        print('Number of tuples:', total_no_tuples)
        print('CAVG:', cavg_raw)
    else:
        print('Number of tuples:', sum(group_length(gr) for gr in groups))
        print('CAVG:', metrics_cavg(groups))    
    no_existing_rules = no_existing_rules / len(R_initial)
    print('Number of existing rules: {}'.format(no_existing_rules))
    no_new_rules, no_loss_rules, no_diff_rules = rules_metrics(R_initial, md_rules)
    print('Number of new rules:', no_new_rules)
    print('Number of loss rules:', no_loss_rules)
    print('Number of diff rules:', no_diff_rules)