"""This file is like example_vary_budget.py but with changes to how the
deterministic baseline is used. In particular, the deterministic baseline now
relaxes the lowers and uppers proportionality constants using the error budget
parameter m. Ultimately, this attempts to evaluate the deterministic algorithm
more generously, by relaxing the proportionality fairness parameters. As a
result 10 choices of m now results in 10 deterministic instances being solved,
whereas previously a single instance was solved.
"""
import configparser
import sys
import os
import numpy as np
import pandas as pd 
import timeit
import random

from collections import defaultdict

from rfc.fair_clustering import fair_clustering
from rfc.util.configutil import read_list
from rfc.util.robustutil import wrapped_get_fairness_violation, worst_fair_vio, flatten_cluster_assignments
from rfc.util.budget import get_valid_lowers_uppers
from rfc.FairType import FairType
from rfc.params import ClusteringParams, RobustParams


if len(sys.argv) <= 2:
    config_file = "../config/example_2_color_config.ini"
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
num_samples = config[config_str].getint("num_samples", None)
seed = config[config_str].getint("seed", 24)
relax_bounds_in_det = config[config_str].getboolean("relax_bounds_in_det", False)
np.random.seed(seed)
random.seed(seed)

# ready up for the loop
df = pd.DataFrame(columns=[
    'num_clusters','Run_Time', 'm', 'lowers',
    'uppers', 'budgetViolation', 'perClusterVio', 'worstPossibleFairVio',
    'cluster_szs', 'additive_violations', 'max_add_violation',
    'objective', 'FairType', 'name', 'num_colors'])
iter_idx = 0 

# Compute lowers/uppers taking prob clustering into acount here
max_errors = max(budgets)
if joint_exps:
    print("Running with joint exps flag: same lower/upper flag will be used for each dataset...")
    assert (lowers == [] and uppers == []), "Have joint exps but manually passed in lowers/uppers."
    color_flags = np.load(f'color_flags/{dataset}.npy')
    if max_points:
        color_flags = color_flags[:max_points]
    colors, color_counts = np.unique(color_flags, return_counts=True)
    reprs = {color: count / len(color_flags) for color, count in zip(colors, color_counts)}
    lowers, uppers = get_valid_lowers_uppers(max(budgets), m_toh, m_hto, reprs, deltas[0])
    if p_acc == -1:  # automatically infer p_acc from m
        p_acc_from_max = 1 - max_errors
    p_other = (1 - p_acc_from_max) / (num_colors - 1)
    for cur_m in budgets:
        cur_prob_vals = np.ones((len(color_flags), num_colors)) * p_other
        for row, color in zip(cur_prob_vals, color_flags):
            row[color] = p_acc_from_max
        prob_vals_reprs = np.mean(cur_prob_vals, axis=0)
        assert len(prob_vals_reprs) == num_colors
        lowers_prob, uppers_prob = get_valid_lowers_uppers(max_errors, m_toh, m_hto, prob_vals_reprs, delta=deltas[0])
        lowers = np.minimum(lowers, lowers_prob)
        uppers = np.maximum(uppers, uppers_prob)
    # Convert them back to list to else json dump errors out
    lowers = lowers.tolist()
    uppers = uppers.tolist()

og_lowers, og_uppers = list(lowers), list(uppers)
for m in budgets:
    lowers, uppers = list(og_lowers), list(og_uppers)
    # Relax the lowers and uppers
    if relax_bounds_in_det and fair_type == FairType.DET:
        for i in range(num_colors):
            # Take a min/max so the proportionality constraints are not vacuous
            lowers[i] = max(lowers[i] - m * cluster, 1 / max_points)
            uppers[i] = min(uppers[i] + m * cluster, 1 - 1 / max_points)

    # infer p_acc from m
    p_acc = 1 - m
    robust_params = RobustParams(
        m_toh=m_toh if m_toh else [1] * num_colors, # array where the i-th position gives m_{\to i}
        m_hto=m_hto if m_hto else [1] * num_colors,
        m=m,
        all_ms=budgets
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
    assert abs(p_acc - (1 - m)) < 1E-3
    start_time = timeit.default_timer()
    output = fair_clustering(clustering_params)
    elapsed_time = timeit.default_timer() - start_time

    x_rounded = output['assignment'] 

    scaling = output['scaling']
    clustering_method = output['clustering_method']
    uppers = output['alpha']
    lowers = output['beta']
    prob_vals = output['prob_values'] 

    num_points = sum(x_rounded)
    ( _ , prob_vals), = prob_vals.items()
    prob_vals = np.array(prob_vals)

    # NOTE: Fairness violations are now computed once per m for deterministic in this file / experiment setup
    # Fairness violations in worst case label realization given budget m.
    # If fair type is not robust, we wish to solve LP once and compute all violations from the corresponding solution
    if fair_type in [FairType.ROBUST, FairType.DET]:
        budget_fairness_violation, per_cluster_vio = wrapped_get_fairness_violation(
            prob_vals,
            x_rounded,
            lowers,
            uppers,
            cluster,
            num_points,
            num_colors,
            m,
            m_hto,
            m_toh,
            fair_type,
        )
    else:  # FairType.PROB
        fairness_vio_results = [
            wrapped_get_fairness_violation(
                prob_vals,
                x_rounded,
                lowers,
                uppers,
                cluster,
                num_points,
                num_colors,
                m,
                m_hto,
                m_toh,
                fair_type,
                num_samples=num_samples,
            )
        ]
        budget_fairness_violation, per_cluster_vio = [], []
        # There are len(budgets) * num_samples fairness violations
        for results in fairness_vio_results:
            bfvs, per_clust_vios = list(zip(*results))
            budget_fairness_violation.extend(bfvs)
            per_cluster_vio.extend(per_clust_vios)

    lp_sol = output['partial_assignment'][:int(cluster*num_points)]
    lp_sol = np.reshape(lp_sol, (-1, cluster))
    # Ensure each point is assigned to 1 cluster
    assert np.all(abs(1 - lp_sol.sum(axis=1)) <= 1E-5)

    # TODO: Does this work for the differently shaped per_cluster_vios from FairType.PROB?
    # Get additive violations and cluster sizes
    flattened_assignments = flatten_cluster_assignments(x_rounded, cluster)
    cluster_szs = [0] * cluster
    for cluster_ix in flattened_assignments:
        cluster_szs[cluster_ix] += 1
    additive_violations = (np.array(cluster_szs) * np.array(per_cluster_vio)).tolist()
    # Need to max over axis=1 if there is >1 set of clusterings
    if fair_type in [FairType.ROBUST, FairType.DET]:
        max_additive_violation = np.max(additive_violations)
    else:
        max_additive_violation = np.max(additive_violations, axis=1)

    # If not robust or prob, write all m values since the experiment only iterates once
    objective = output["objective"]
    name = output["name"]
    # TODO: Add said cluster sizes to the baseline and other dataframes
    m_to_write = m
    if fair_type == FairType.ROBUST:
        worst_possible_fair_vio = output["worst_possible_fairness_vio"]
    else:
        worst_possible_fair_vio = worst_fair_vio(lowers, uppers)
        # if fair_type == FairType.DET:
        #     m_to_write = budgets
        #     bfv_budgets = budgets
        # else:
        # bfv_budgets = np.repeat(budgets, num_samples)
        # k = len(budget_fairness_violation)
        # if fair_type == FairType.DET:
        #     bfv_df = pd.DataFrame({
        #             'm': bfv_budgets,
        #             'budgetViolation': budget_fairness_violation,
        #             'per_cluster_vio': per_cluster_vio,
        #             'worstPossibleFairVio': worst_possible_fair_vio,
        #             'cluster_szs': [cluster_szs] * len(budgets),
        #             'additive_violations': additive_violations,
        #             'max_add_violation': max_additive_violation,
        #             'objective': objective,
        #             'FairType': [fair_type] * k,
        #             'name': name,
        #             'num_colors': num_colors
        #             })

    # TODO: Add the additive fairness violation and cluster size statistics to these dataframes
    if isinstance(budget_fairness_violation, list):
        for i in range(len(budget_fairness_violation)):
            df.loc[iter_idx] = [
                cluster,elapsed_time,
                m_to_write, lowers, uppers, budget_fairness_violation[i], per_cluster_vio[i], worst_possible_fair_vio,
                cluster_szs, additive_violations[i], max_additive_violation[i],
                objective, fair_type, name, num_colors
                ]
            iter_idx += 1
    else:
        df.loc[iter_idx] = [
            cluster,elapsed_time,
            m_to_write, lowers, uppers, budget_fairness_violation, per_cluster_vio, worst_possible_fair_vio,
            cluster_szs, additive_violations, max_additive_violation,
            objective, fair_type, name, num_colors
            ]

    iter_idx += 1 

scale_flag = 'normalized' if scaling else 'unnormalized' 
p_acc_str = 'p' + str(p_acc-int(p_acc))[2:]
filename = (
    "_".join(
        [
            exp_name,
            dataset,
            clustering_method,
            str(int(num_points)),
            scale_flag,
            p_acc_str,
        ]
    )
    + ".csv"
)

# do not over-write
exp_folders = ['Results', 'BaselineViolations', 'ErrModelsViolations']
paths = {p: os.path.join('experiments', exps_family, p) for p in exp_folders}
filepath = os.path.join(paths['Results'], filename)
while os.path.isfile(filepath):
    filename ='new' + filename 
    filepath = os.path.join(paths['Results'], filename)
for p in paths.values():
    if not os.path.exists(p):
        os.makedirs(p)
df.to_csv(filepath, sep=',',index=False)
# NOTE: No baseline violations files, unlike in example_vary_budget where a
# single instance is solved and violations are computed using all values of m in
# this single instance's solution
# print('About to write baselineviolations')
# if fair_type == FairType.DET:
#     bfv_df.to_csv(os.path.join(paths['BaselineViolations'], filename), sep=',', index=False)
