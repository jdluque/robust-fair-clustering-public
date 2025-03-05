import configparser
import time
import random
from collections import defaultdict
import numpy as np

from . import params

from .cplex_fair_assignment_lp_solver import fair_partial_assignment
from .util.clusteringutil import (
    clean_data,
    read_data,
    scale_data,
    vanilla_clustering,
    write_fairness_trial,
)
from .util.configutil import read_list
from .util.probutil import (
    create_prob_vecs,
)
from .util.robustutil import (
    worst_fair_vio,
)
from .util.budget import get_valid_lowers_uppers
from .FairType import FairType


def fair_clustering(clustering_params: params.ClusteringParams):
    """This function takes a dataset and performs a fair clustering on it.

    Arguments:
    dataset (str) : dataset to use
    config_file (str) : config file to use (will be read by ConfigParser)
    data_dir (str) : path to write output
    num_clusters (int) : number of clusters to use
    deltas (list[float]) : delta to use to tune alpha, beta for each color
    max_points (int ; default = 0) : if the number of points in the dataset 
        exceeds this number, the dataset will be subsampled to this amount.
    fair_type (FairType(Enum)): type of fair clustering to run  
    correlated_scramble (bool): whether to mess up the points in a correlated
        manner. This is as discussed and for the sake of showcasing probailistic
        clustering's shortcoming.
    robust_params: paramters for robust fair clustering experiments
    Output:
    None (Writes to file in `data_dir`)  
    """
    dataset = clustering_params.dataset
    config_file = clustering_params.config_file
    data_dir = clustering_params.data_dir
    num_clusters = clustering_params.num_clusters
    deltas = clustering_params.deltas
    max_points = clustering_params.max_points
    L = clustering_params.L
    p_acc = clustering_params.p_acc
    fair_type = clustering_params.fair_type
    lowers = clustering_params.lowers
    uppers = clustering_params.uppers
    robust_params = clustering_params.robust_params
    # Set the seed once in, e.g., example_vary_budget.py
    # seed = clustering_params.seed
    # np.random.seed(seed)
    # random.seed(seed)

    if type(fair_type) != FairType:
        raise ValueError(f'Invalid fair_type={fair_type}, should be one of {list(FairType)}')

    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # Read data in from a given csv_file found in config
    if clustering_params.test_df is None:
        df = read_data(config, dataset)
    else:
        df = clustering_params.test_df
    # Subsample data if needed
    if max_points and len(df) > max_points:
        # NOTE: comment the block and second and unccomment the second block. changed to exclude randomization effect
        # rows = [0,1,2,3,4,5,20,21,23,50,126,134,135]
        # df = df.iloc[rows,:]
        # df = df.reset_index()
        df = df.head(max_points)
        # below if you wish to shuffle
        # df= df.sample( frac=1, random_state=1).reset_index(drop=True)

    # Clean the data (bucketize text data)
    if clustering_params.test_df is None:
        df, _ = clean_data(df, config, dataset)
    # Automatically set p_acc in proportion to m, to compare with robust experiments
    # Fix robust params
    if fair_type == FairType.ROBUST:
        # Round m up number which multiplies with the number of points to give an integer
        robust_params.m = len(df) * robust_params.m
        m = robust_params.m
        robust_params.m_toh  = [val * len(df) for val in robust_params.m_toh]
        robust_params.m_hto  = [val * len(df) for val in robust_params.m_hto]
        robust_params.m_toh  = [np.ceil(min(m, val)) for val in robust_params.m_toh]
        robust_params.m_hto  = [np.ceil(min(m, val)) for val in robust_params.m_hto]
    m_toh = robust_params.m_toh
    m_hto = robust_params.m_hto
    # variable_of_interest (list[str]) : variables that we would like to collect statistics for
    variable_of_interest = config[dataset].getlist("fairness_variable")
    # NOTE: this code only handles one color per vertex
    assert len(variable_of_interest) == 1 
    var = variable_of_interest[0]

    # Assign each data point to a color, based on config file
    # attributes (dict[str -> defaultdict[int -> list[int]]]) : holds indices of points for each color class
    # color_flag (dict[str -> list[int]]) : holds map from point to color class it belongs to (reverse of `attributes`)
    attributes, color_flag, prob_vecs, prob_vals, prob_vals_thresh, prob_thresh = {}, {}, {}, {}, {}, {}
    for variable in variable_of_interest:
        if clustering_params.test_df is not None:
            break
        colors = defaultdict(list)
        this_color_flag = [0] * len(df)

        condition_str = variable + "_conditions"
        bucket_conditions = config[dataset].getlist(condition_str)

        # For each row, if the row passes the bucket condition, then the row is
        # added to that color class
        for i, row in df.iterrows():
            for bucket_idx, bucket in enumerate(bucket_conditions):
                if eval(bucket)(row[variable]):
                    colors[bucket_idx].append(i)  # add the point to the list of its colors 
                    this_color_flag[i] = bucket_idx  # record the color for this given point 

        attributes[variable] = colors     
        color_flag[variable] = this_color_flag

        # NOTE: generate probabilities according to the perturbation descired in section 5.2
        # Make prob_vals a matrix
        if fair_type == FairType.PROB:
            prob_vals[variable] = create_prob_vecs(len(df), p_acc, len(colors), this_color_flag)
        else:
            prob_vals[variable] = np.zeros((len(df), len(colors)))
            for row, color in zip(prob_vals[variable], this_color_flag):
                row[color] = 1

    # NOTE: Uncomment this block to write the color_flag to ./color_flags/{dataset_name}.npy and exits the code
    # assert max_points == 0  # ensure w=color flag is being written for full dataset
    # np.save(f'color_flags/{dataset}.npy', this_color_flag)
    # assert False

    # representation (dict[str -> dict[int -> float]]) : representation of each color compared to the whole dataset
    if clustering_params.test_df is None:
        representation = {}
        for var in variable_of_interest:
            representation[var] = np.mean(prob_vals[var], axis=0)
            assert abs(sum(representation[var]) - 1) < 1E-4
            representation[var] = {color: repres_val for color,repres_val in enumerate(representation[var])}

        ( _ , fair_vals), = representation.items()
        # NOTE: this handles the case when a color is missing in the sampled vertices
        num_colors = max(fair_vals.keys()) + 1

        # drop uneeded columns
        selected_columns = config[dataset].getlist("columns")
        df = df[[col for col in selected_columns]]

        # Scale data if desired
        scaling = config["DEFAULT"].getboolean("scaling")
        if scaling:
            df = scale_data(df)
    else:
        # in test mode: must manually set number of colors
        num_colors = len(clustering_params.lowers)

    # Cluster the data -- using the objective specified by clustering_method
    clustering_method = config["DEFAULT"]["clustering_method"]
    print("clustering_method: ", clustering_method)

    t1 = time.monotonic()
    # NOTE: initial_score is the value of the objective at the solution
    # NOTE: This is where the color-blind algorithm is ran
    if fair_type != FairType.ROBUST:
        if type(num_clusters) is list:
            num_clusters = num_clusters[0] 
        initial_score, pred, cluster_centers = vanilla_clustering(df, num_clusters, clustering_method)
    else:
        initial_score, pred, cluster_centers = -1, [0] * num_clusters, [[0]] * num_clusters 
        print(cluster_centers)
        print(len(cluster_centers))
        print('Running robust clustering: skipping vanilla clustering')
    t2 = time.monotonic()
    cluster_time = t2 - t1
    print("Clustering time: {}".format(cluster_time))

    # sizes (list[int]) : sizes of clusters
    print('clust_centers: ', cluster_centers)
    sizes = [0 for _ in range(num_clusters)]
    for p in pred:
        sizes[p] += 1
    print('sizes: ', sizes)

    # fairness_vars (list[str]) : Variables to perform fairness balancing on
    fairness_vars = config[dataset].getlist("fairness_variable")

    # NOTE: here is where you set the upper and lower bounds
    # NOTE: accross all different values within the same attribute you have the same multipliers up and down
    # ========================================
    # Behavior for selecting upper / lower proportionality values
    # ========================================
    # If max_m is not in robust_params, set the values in the old manner: for
    # each color h, alpha_h [beta_h] is set to
    # repr[h] * (1 - delta) [repr[h] / # (1 - delta)] where repr[h] is the
    # fraction of the dataset with label h
    for delta in deltas:
        if lowers != [] and uppers != []:
            # Avoid automatically setting lowers and uppers
            print(f"Manually passed in lowers={lowers} and uppers={uppers}")
            pass
        # This is left as the legacy way of setting lower and upper
        # proportionality demands.
        elif 'max_m' not in robust_params:
            lowers, uppers = [], []
            a_val, b_val = 1 / (1 - delta) , 1 - delta
            # NOTE: 2 color case
            # NOTE: repr gives repr of color=1, want color_repr to ensure the below condition is met for all colors
            for color in range(num_colors):
                for var, bucket_dict in attributes.items():
                    cur_alpha = a_val*representation[var][color]
                    cur_beta = b_val*representation[var][color]
                    # NOTE: sanity checks for upper (alpha) and lower (beta) bounds; otherwise problem becomes infeasbile
                    if cur_alpha < representation[var][color] + m_toh[color] / len(df): 
                        new_val = representation[var][color] + m_toh[color] / len(df)
                        print(f'Had to grow u_{color} to from', cur_alpha, ' to ', new_val)
                        cur_alpha = new_val
                    if cur_beta > representation[var][color] - m_hto[color] / len(df):
                        new_val = representation[var][color] - m_hto[color] / len(df)
                        print(f'Had to grow l_{color} to from', cur_beta, ' to ', new_val)
                        cur_beta = new_val
                lowers.append(cur_beta)
                uppers.append(cur_alpha)
        # Set lowers and uppers to the tightest possible values manually
        # TODO: Want to move this code checking for a common uppers lowers to something like example_vary_budget.py
        else:
            # representation from deterministic color assignments
            reprs = []
            for var in representation:
                for color in range(num_colors):
                    reprs.append(representation[var][color])
            frac_m_toh = np.array(m_toh) / len(df)
            frac_m_hto = np.array(m_hto) / len(df)
            # NOTE: take max_m and unshrunk m_to_h vectors so that this leads
            # the loosest possible bounds among all experiments that will be
            # run.
            lowers, uppers = get_valid_lowers_uppers(robust_params.max_m, m_toh, m_hto, reprs, delta=delta)
            print(f'reprs: {reprs}')
            print('Automatically set lowers, uppers: ')
        print('lowers, uppers: ', lowers, uppers)
        fp_color_flag = prob_vals

        # Whether to use probabilistic or deterministic color flags
        # NOTE: Want deterministic color_flags for deterministic and robust and
        #       probabilistic for probabilistic and robust
        fair_type_color_flag = fp_color_flag if fair_type == FairType.PROB else color_flag
        if clustering_params.test_df is not None:
            # var is the fairness "variable of interest"
            fair_type_color_flag = {var: clustering_params.test_colors}
        # Solves partial assignment and then performs rounding to get integral assignment
        print('ftcf: ', fair_type_color_flag)
        t1 = time.monotonic()
        res = fair_partial_assignment(df, cluster_centers, uppers, lowers, fair_type_color_flag, clustering_method, num_colors, L, fair_type, robust_params)
        t2 = time.monotonic()
        lp_time = t2 - t1

        ### Output / Writing data to a file
        # output is a dictionary which will hold the data to be written to the
        #   outfile as key-value pairs. Outfile will be written in JSON format.
        output = {}
        # NOTE: fairness violations being computed in example_vary_budget.py
        output["worst_possible_fairness_vio"] = worst_fair_vio(lowers, uppers)

        # Record Assignments
        output["partial_assignment"] = res["partial_assignment"]
        output["assignment"] = res["assignment"]

        # Clustering score after addition of fairness
        output["objective"] = res["objective"]

        # Clustering score after initial LP
        output["partial_objective"] = res["partial_objective"]

        output['prob_values'] = prob_vals
        if clustering_params.test_df is None:
            # num_clusters for re-running trial
            output["num_clusters"] = num_clusters

            # Whether or not the LP found a solution
            output["partial_success"] = res["partial_success"]

            # Nonzero status -> error occurred
            output["partial_status"] = res["partial_status"]

            # output["dataset_distribution"] = dataset_ratio

            # Save alphas and betas from trials
            output['prob_proportions'] = representation
            output["alpha"] = uppers
            output["beta"] = lowers

            # Save size of each cluster
            output["sizes"] = sizes

            output["attributes"] = attributes

            # These included at end because their data is large
            # Save points, colors for re-running trial
            # Partial assignments -- list bc. ndarray not serializable
            output["centers"] = [list(center) for center in cluster_centers]
            output["points"] = [list(point) for point in df.values]

            # Save original clustering score
            output["unfair_score"] = initial_score
            # Original Color Blind Assignments
            if type(pred) is not list:
                pred = pred.tolist() 

            output["unfair_assignments"] = pred 


            # Record Lower Bound L
            output['Cluster_Size_Lower_Bound'] = L

            # Record Classifier Accurecy
            output['p_acc'] = p_acc

            # Record probability vecs
            output["name"] = dataset
            output["clustering_method"] = clustering_method
            output["scaling"] = scaling
            output["delta"] = delta
            output["time"] = lp_time
            output["cluster_time"] = cluster_time


            output["pof"] = output["objective"] / output["unfair_score"]
            output["fair_type"] = str(fair_type)
            output["m_toh"] = m_toh
            output["m_hto"] = m_hto
            # Writes the data in `output` to a file in data_dir
            write_fairness_trial(output, data_dir)
        # Added because sometimes the LP for the next iteration solves so
        # fast that `write_fairness_trial` cannot write to disk
        time.sleep(1) 

        print('objective ', output["objective"])

        return output  
