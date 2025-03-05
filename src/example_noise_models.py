import configparser
import sys
import os
import numpy as np
import pandas as pd 
import timeit
import itertools as it
from pathlib import Path
from collections import defaultdict


from rfc.fair_clustering import fair_clustering
from rfc.util.configutil import read_list
from rfc.util.utilhelpers import (max_Viol, x_for_colorBlind, find_balance,
        find_proprtions_two_color,
        max_Viol_multi_color, prob_vec_two_to_multi_color, find_proprtions_multi_color,
        find_balance_multi_color)
from rfc.util.robustutil import wrapped_get_fairness_violation, worst_fair_vio
from rfc.util.clusteringutil import read_data, clean_data, scale_data
from rfc.util.probutil import create_prob_vecs
from rfc.util.budget import get_valid_lowers_uppers
from rfc.FairType import FairType
from rfc.params import ClusteringParams, RobustParams


if len(sys.argv) <= 2:
    config_file = "config/example_2_color_config.ini"
else:
    config_file = sys.argv[2]
print(config_file)
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

# Create your own entry in `example_config.ini` and change this str to run
# your own trial
# bank_binary_marital 
config_str = "bank_binary_marital_budget_plot" if len(sys.argv) == 1 else sys.argv[1]
print(f'config_str: {config_str}')


# Read variables
data_dir = config[config_str].get("data_dir")
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")
p_acc = float(config[config_str].get("p_acc", -1)) 
# Ts = list(map(float, config[config_str].getlist("Ts")))
# TODO: Finish implementing experiments for probabilistic and deterministic
fair_type = eval(str(config[config_str].get("fair_type")))
cluster = int(config[config_str].get("num_clusters"))
budgets = eval(str(config[config_str].get("budgets")))
exp_name = str(config[config_str].get("exp_name", ""))
# Taking this parameter to shape the size of m_to_h and m_h_to
num_colors = config[config_str].getint("num_colors")
m_toh = list(map(float, config[config_str].getlist("m_toh", np.ones(num_colors))))
m_hto = list(map(float, config[config_str].getlist("m_hto", np.ones(num_colors))))
lowers = list(map(float, config[config_str].getlist("lowers", [])))
uppers = list(map(float, config[config_str].getlist("uppers", [])))
joint_exps = config[config_str].getboolean("joint_exps", False)
exps_family = config[config_str].get("exps_family", "")
noise_model_contrast = config[config_str].getboolean("noise_model_contrast", False)
noise_model = config[config_str].get("noise_model")

# ready up for the loop 
df = pd.DataFrame(columns=[
    'num_clusters','POF','MaxViolFair','MaxViolUnFair','Fair Balance', 
    'UnFair Balance','Run_Time','ColorBlindCost','FairCost', 'm', 'lowers',
    'uppers', 'budgetViolation', 'perClusterVio', 'worstPossibleFairVio',
    'objective', 'FairType', 'name', 'num_colors'])
iter_idx = 0 

max_errors = max(budgets)
max_m_toh = np.minimum(m_toh, max_errors)
max_m_hto = np.minimum(m_hto, max_errors)
# Don't use this line because this value of m will be used in bae
# max_errors = min(max_errors, max(max(m_hto), max(m_toh)))
if joint_exps:
    print("Running with joint exps flag: same lower/upper flag will be used for each dataset...")
    assert (lowers == [] and uppers == []), "Have joint exps but manually passed in lowers/uppers."
    color_flags = np.load(f'color_flags/{dataset}.npy')
    if max_points:
        color_flags = color_flags[:max_points]
    colors, color_counts = np.unique(color_flags, return_counts=True)
    reprs = {color: count / len(color_flags) for color, count in zip(colors, color_counts)}
    print('reprs: ', reprs)
    # largest errors possible happen under bae model where m_toh and m_hto are all 1's
    lowers, uppers = get_valid_lowers_uppers(max_errors, [1] * num_colors, [1] * num_colors, reprs, deltas[0])
    print(lowers, uppers)
    print(max_errors)
    print(max_m_toh, m_hto)

robust_vios_df = pd.DataFrame(columns=[
    'm', 'noise_model', 'violation', 'objective', 'num_colors',
    'per_cluster_vio', 'worst_possible_fair_vio', 'fair_type', 'dataset',
    'lowers', 'uppers'
])
print(budgets)
# noise_models = ['equally_noisy', 'color_0_gains_more', 'color_1_gains_more']
# If not robust, then only run the experiments once
for m in budgets:
    m_toh, m_hto = list(max_m_toh), list(max_m_hto)
    if p_acc == -1:  # automatically infer p_acc from m
        p_acc = 1 - max_errors
    # Adjust error paramters depending on error model
    if noise_model == 'equally_noisy':
        m_toh, m_hto = [1] * num_colors, [1] * num_colors
    # NOTE: Must change for >2 colors
    elif noise_model == 'color_0_gains_more':
        m_toh, m_hto = [m, m/2], [m/2, m]
    elif noise_model == 'color_1_gains_more':
        m_toh, m_hto = [m/2, m], [m, m/2]
    else:
        raise Exception('Unexpected noise model.')
    robust_params = RobustParams(
        m_toh=m_toh,
        m_hto=m_hto,
        m=m,
        all_ms=budgets,
    )
    clustering_params = ClusteringParams(
        dataset,
        clustering_config_file,
        data_dir,
        cluster,
        deltas,
        max_points,
        0,
        p_acc,
        lowers=lowers,
        uppers=uppers,
        fair_type=fair_type,
        robust_params=robust_params,
    )

    print('max_m: ', robust_params.max_m)
    start_time = timeit.default_timer()
    output = fair_clustering(clustering_params)
    elapsed_time = timeit.default_timer() - start_time


    fair_cost = output['objective']
    if fair_type != FairType.ROBUST:
        colorBlind_cost = output['unfair_score']
        POF = fair_cost/colorBlind_cost
        x_color_blind = x_for_colorBlind(output['unfair_assignments'],cluster)
    else:
        # for robust, fill in those values with placeholders
        colorBlind_cost, POF, x_color_blind = 3 * [-1]
    x_rounded = output['assignment'] 

    scaling = output['scaling']
    clustering_method = output['clustering_method']
    uppers = output['alpha']
    lowers = output['beta']
    prob_vals = output['prob_values'] 

    num_points = sum(x_rounded)
    # Robust has no vanilla asignments so ignore this check then
    if fair_type != FairType.ROBUST:
        assert sum(x_rounded)==sum(x_color_blind)

    # ( _ , alpha), = alpha.items()
    # ( _ , beta), = beta.items()
    ( _ , prob_vals), = prob_vals.items()
    prob_vals = np.array(prob_vals)

    # maxViolFair = max_Viol(x_rounded,num_colors,prob_vals, cluster,alpha,beta)
    # Computes fairness violations WITHOUT using the perturbed labels budget m
    if fair_type != FairType.ROBUST:
        maxViolFair = max_Viol_multi_color(x_rounded, num_colors, prob_vals, cluster, uppers, lowers)
        maxViolUnFair = max_Viol_multi_color(x_color_blind, num_colors, prob_vals, cluster, uppers, lowers)
    else:
        # fill in placeholder vals for robust
        maxViolFair, maxViolUnFair = -1, -1

    proportion_data_set = sum(prob_vals)/num_points 
    if fair_type != FairType.ROBUST:
        fair_balance = find_balance_multi_color(x_rounded,num_colors, cluster,prob_vals,proportion_data_set)
        unfair_balance = find_balance_multi_color(x_color_blind,num_colors, cluster,prob_vals,proportion_data_set)
    else:
        # TODO: fill in placeholder vals for robust
        fair_balance, unfair_balance = -1, -1

    # Fairness violations in worst case label realization given budget m
    # If fair type is not robust, we wish to solve LP once and compute all violations from the corresponding solution
    # TODO: Record fairness violations for each error model here
    worst_possible_fair_vio = worst_fair_vio(lowers, uppers)
    if fair_type == FairType.ROBUST:
        budget_fairness_violation, per_cluster_vio = wrapped_get_fairness_violation(prob_vals, x_rounded, lowers, uppers, cluster, num_points, num_colors, m, m_hto_frac=m_hto, m_toh_frac=m_toh, fair_type=fair_type)
        robust_vios_df.loc[len(robust_vios_df)] = [
            m, noise_model, budget_fairness_violation, output["objective"], num_colors,
            per_cluster_vio, worst_possible_fair_vio, fair_type, dataset, lowers, uppers]

    lp_sol = output['partial_assignment'][:int(cluster*num_points)]
    lp_sol = np.reshape(lp_sol, (-1, cluster))
    # Ensure each point is assigned to 1 cluster 
    assert np.all(lp_sol.sum(axis=1) <= 1 + 1E-5)

    # _, props , sizes= find_proprtions_two_color(x_rounded,num_colors,prob_vals,cluster)
    _, props , sizes= find_proprtions_multi_color(x_rounded,num_colors,prob_vals,cluster)


    # If not robust, write all Ts and ms since the experiment only iterates once
    objective = output["objective"]
    name = output["name"]
    if fair_type == FairType.ROBUST:
        m_to_write = m
        worst_possible_fair_vio = output["worst_possible_fairness_vio"]

    df.loc[iter_idx] = [
        cluster,POF,maxViolFair,maxViolUnFair,fair_balance,unfair_balance,elapsed_time,colorBlind_cost,fair_cost,
        m_to_write, lowers, uppers, budget_fairness_violation, per_cluster_vio,
        worst_possible_fair_vio, objective, fair_type, name, num_colors
        ]

    iter_idx += 1 



scale_flag = 'normalized' if scaling else 'unnormalized' 
filename = exp_name + '_' + noise_model + '_' + dataset + '_' + clustering_method + '_' + str(int(num_points)) + '_' + scale_flag  
p_acc_str = 'p' + str(p_acc-int(p_acc))[2:]
filename = filename + '_' + p_acc_str
filename = filename + '.csv'



# do not over-write 
results_folder = f'experiments/{exps_family}/Results'
baselines_folder = f'experiments/{exps_family}/BaselineViolations'
robust_vios_folder = f'experiments/{exps_family}/ErrModelsViolations' 
filepath = Path(results_folder + '/'+ filename)
while filepath.is_file():
    filename='new' + filename 
    filepath = Path(results_folder + '/'+ filename)

paths = [results_folder, baselines_folder, robust_vios_folder]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)
df.to_csv(results_folder + '/'+ filename, sep=',',index=False)
if noise_model_contrast:
    robust_vios_df.to_csv(f'{robust_vios_folder}/{filename}', sep=',', index=False)
