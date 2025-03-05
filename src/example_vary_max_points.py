import configparser
import sys
import os
import numpy as np
import pandas as pd 
import timeit
import random

from rfc.fair_clustering import fair_clustering
from rfc.util.configutil import read_list
from rfc.util.robustutil import wrapped_get_fairness_violation, worst_fair_vio
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
max_pointss = list(map(int, config[config_str].getlist("max_pointss")))
assert len(max_pointss) == 3
max_pointss = range(*max_pointss)
# max_points = config[config_str].getint("max_points")
p_acc = float(config[config_str].get("p_acc", -1)) 
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
np.random.seed(seed)
random.seed(seed)

# ready up for the loop
df = pd.DataFrame(columns=[
    'max_points', 'num_clusters','Run_Time', 'm', 'lowers',
    'uppers', 'budgetViolation', 'perClusterVio', 'worstPossibleFairVio',
    'objective', 'FairType', 'name', 'num_colors'])
iter_idx = 0 

# Compute lowers/uppers taking the varying max_points into account
# TODO: Since we are fixing m it suffices to consider a single choice of p_acc and a single choice of lowers/uppers from rfc feasibility constraints
max_errors = max(budgets)
if joint_exps:
    print("Running with joint exps flag: same lower/upper flag will be used for each dataset...")
    assert (lowers == [] and uppers == []), "Have joint exps but manually passed in lowers/uppers."
    # TODO: Get the color flags for the diabetes dataset
    color_flags = np.load(f'color_flags/{dataset}.npy')
    print(max_pointss)
    for subsample_max_points in max_pointss:
        subsample_color_flags = color_flags[:subsample_max_points]
        colors, color_counts = np.unique(subsample_color_flags, return_counts=True)
        reprs = {color: count / len(subsample_color_flags) for color, count in zip(colors, color_counts)}
        lowers, uppers = get_valid_lowers_uppers(max(budgets), m_toh, m_hto, reprs, deltas[0])
        if p_acc == -1:  # automatically infer p_acc from m
            p_acc_from_max = 1 - max_errors
        p_other = (1 - p_acc_from_max) / (num_colors - 1)
        for cur_m in budgets:
            cur_prob_vals = np.ones((len(subsample_color_flags), num_colors)) * p_other
            for row, color in zip(cur_prob_vals, subsample_color_flags):
                row[color] = p_acc_from_max
            prob_vals_reprs = np.mean(cur_prob_vals, axis=0)
            assert len(prob_vals_reprs) == num_colors
            lowers_prob, uppers_prob = get_valid_lowers_uppers(max_errors, m_toh, m_hto, prob_vals_reprs, delta=deltas[0])
            lowers = np.minimum(lowers, lowers_prob)
            uppers = np.maximum(uppers, uppers_prob)
    # Convert them back to list to else json dump errors out
    lowers = lowers.tolist()
    uppers = uppers.tolist()

m = budgets[0]
assert len(budgets) == 1
for max_points in max_pointss:
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

    # Fairness violations in worst case label realization given budget m.
    # If fair type is not robust, we wish to solve LP once and compute all violations from the corresponding solution
    if fair_type == FairType.ROBUST:
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
    elif fair_type == FairType.DET:
        fairness_vio_results = [
            wrapped_get_fairness_violation(
                prob_vals,
                x_rounded,
                lowers,
                uppers,
                cluster,
                num_points,
                num_colors,
                cur_m,
                m_hto,
                m_toh,
                fair_type,
                num_samples=num_samples,
            )
            for cur_m in budgets
        ]
        # Deterministic clustering gives a budget_fairness violation value for each choice in `budgets`. Probabilistic
        # clustering gives num_samples-many violations for each choice of budget.
        if fair_type == FairType.DET:
            budget_fairness_violation, per_cluster_vio = list(zip(*fairness_vio_results))
        else:
            budget_fairness_violation, per_cluster_vio = [], []
            # There are len(budgets) * num_samples fairness violations
            for results in fairness_vio_results:
                bfvs, per_clust_vios = list(zip(*results))
                budget_fairness_violation.extend(bfvs)
                per_cluster_vio.extend(per_clust_vios)
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
                cur_m,
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

    # If not robust or prob, write all m values since the experiment only iterates once
    objective = output["objective"]
    name = output["name"]
    if fair_type == FairType.ROBUST:
        m_to_write = m
        worst_possible_fair_vio = output["worst_possible_fairness_vio"]
    else:
        worst_possible_fair_vio = worst_fair_vio(lowers, uppers)
        if fair_type == FairType.DET:
            m_to_write = budgets
            bfv_budgets = budgets
        else:
            m_to_write = m
            bfv_budgets = np.repeat(budgets, num_samples)
        k = len(budget_fairness_violation)
        if fair_type == FairType.DET:
            bfv_df = pd.DataFrame({
                    'm': bfv_budgets,
                    'budgetViolation': budget_fairness_violation,
                    'per_cluster_vio': per_cluster_vio,
                    'worstPossibleFairVio': worst_possible_fair_vio,
                    'objective': objective,
                    'FairType': [fair_type] * k,
                    'name': name,
                    'num_colors': num_colors
                    })

    if isinstance(budget_fairness_violation, list):
        for i in range(len(budget_fairness_violation)):
            df.loc[iter_idx] = [
                max_points, cluster,elapsed_time,
                m_to_write, lowers, uppers, budget_fairness_violation[i], per_cluster_vio[i],
                worst_possible_fair_vio, objective, fair_type, name, num_colors
                ]
            iter_idx += 1
    else:
        df.loc[iter_idx] = [
            max_points, cluster,elapsed_time,
            m_to_write, lowers, uppers, budget_fairness_violation, per_cluster_vio,
            worst_possible_fair_vio, objective, fair_type, name, num_colors
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
print('About to write baselineviolations')
if fair_type == FairType.DET:
    bfv_df.to_csv(os.path.join(paths['BaselineViolations'], filename), sep=',', index=False)
