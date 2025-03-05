"""Tests for robust fair clustering.
"""

import pytest
import copy
import numpy as np
import pandas as pd
import configparser
from rfc.util.configutil import read_list
from rfc.util.probutil import perturb_2_color
from rfc.util.robustutil import (get_correlated_scramble, scrambled_representations,
      scramble_statistics, get_fairness_violation)
from rfc.fair_clustering import fair_clustering
from rfc.FairType import FairType

import rfc.util.robustutil as ru
import rfc.params


def assignment_vector(assignment_vector, num_clusters):
    return np.argmax(np.array(assignment_vector).reshape((-1, num_clusters)), axis=1)

@pytest.fixture
def clustering_params():
    """Used to build test_small_robust_instance and other tests. Must set lowers
    and uppers.
    """
    config_file = "config/example_2_color_config.ini"
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # Create your own entry in `example_config.ini` and change this str to run
    # your own trial
    config_str = "test_bank_binary_marital"

    # Read variables
    data_dir = config[config_str].get("data_dir")
    dataset = config[config_str].get("dataset")
    clustering_config_file = config[config_str].get("config_file")
    num_clusters = int(config[config_str].get("num_clusters"))
    deltas = list(map(float, config[config_str].getlist("deltas")))
    max_points = config[config_str].getint("max_points")
    budgets = eval(str(config[config_str].get("budgets")))
    p_acc = float(config[config_str].get("p_acc")) 
    fair_type = eval(str(config[config_str].get("fair_type")))
    num_colors = config[config_str].getint("num_colors")
    robust_params = rfc.params.RobustParams(
        m_toh= [1] * num_colors, # array where the i-th position gives m_{\to i}
        m_hto= [1] * num_colors,
        m=budgets[0],
        all_ms=budgets
    )
    clustering_params = rfc.params.ClusteringParams(
        dataset,
        clustering_config_file,
        data_dir,
        num_clusters,
        deltas,
        max_points,
        0,
        p_acc,
        lowers=None,
        uppers=None,
        fair_type=fair_type,
        robust_params=robust_params,
    )
    return clustering_params

def test_bank_testset(clustering_params):
    """Ensure that robust fair clustering can assign all points to the same
    cluster as required. In this example instance, there are 8 points, equally
    sharing 2 colors, that must be placed into two clusters. However, m=0.25 so
    2 points will be misclassified; leaving the only valid assignment to be
    placing all points into a single cluster.
    
    Adapted contents from `example_2_color.py`.
    """
    lowerss = [
        [0.25, 0.25],
        [0.01, 0.01],
    ]
    upperss = [
        [0.75, 0.75],
        [.99, .99],
    ]
    for lowers, uppers, in zip(lowerss, upperss):
        num_colors = len(lowers)
        # Deep copy necessary because fair_clustering_2_color() updates robust_params 
        cp = copy.deepcopy(clustering_params)
        cp.lowers = lowers
        cp.uppers = uppers
        output = fair_clustering(cp)
        # Cost of rounded solution should be 0 because points 1 and 2 belong to
        # cluster 1 and points 3 and 4 belong to cluster 2, and that is how they are
        # assigned
        colors = output['attributes']['marital']
        colors_flag = [0] * cp.max_points
        for j in colors[1]:
            colors_flag[j] = 1
        assignments = assignment_vector(output['assignment'], cp.num_clusters)
        print(assignments)
        cluster_color_map = ru.cluster_color_to_point(assignments, colors_flag, 2)
        cluster_szs = [sum(len(cluster_color_map[i, j]) for j in range(num_colors))
                        for i in range(cp.num_clusters)]
        for cluster in range(cp.num_clusters):
            if cluster_szs[cluster]:
                for color in range(num_colors):
                    proportion = len(cluster_color_map[cluster, color]) / cluster_szs[cluster]
                    assert lowers[color] <= proportion  <= uppers[color]
        # the k-center objective: maximum distance between a point and its assigned center
        print(output["objective"])
        assert output["objective"] >= 0

def test_small_dataframe(clustering_params):
    """Test on a small instance with 12 points, 3 colors, up to 4 clusters, and
    error budget = 1. We set the number of clusters. The only feasible solutions
    are assigning the points to one or two clusters. The oprtimal solution is
    using two clusters.
    """
    cp = clustering_params
    num_colors = 3
    cp.robust_params.m_hto = [1] * num_colors
    cp.robust_params.m_toh = [1] * num_colors
    cp.robust_params.m = 1/12
    cp.robust_params.all_ms = [cp.robust_params.m]
    cp.test_df = pd.DataFrame.from_records([
        [1],
        [1],
        [2],
        [2],
        [2],
        [2],
        [10],
        [10],
        [10],
        [10],
        [11],
        [11],
    ])
    # 1D array of colors: i-th entry gives color of i-th point
    cp.test_colors = [
        0,
        0,
        1,
        1,
        2,
        2,
        1,
        1,
        2,
        2,
        0,
        0,
    ]
    eps = .1
    cp.lowers = [1/6 - eps, 1/6 - eps, 1/6 - eps]
    cp.uppers = [1/2 + eps, 1/2 + eps, 1/2 + eps]
    cp.num_clusters = 4
    cp.max_points = 12
    output = fair_clustering(cp)
    # assignments = np.argmax(np.array(output['assignment']).reshape((cp.max_points, cp.num_clusters)), axis=1)
    assignments = assignment_vector(output['assignment'], cp.num_clusters)
    cluster_color_map = ru.cluster_color_to_point(assignments, cp.test_colors, num_colors)
    cluster_szs = [sum(len(cluster_color_map[i, j]) for j in range(num_colors))
                    for i in range(cp.num_clusters)]
    for cluster in range(cp.num_clusters):
        if cluster_szs[cluster]:
            for color in range(num_colors):
                proportion = len(cluster_color_map[cluster, color]) / cluster_szs[cluster]
                assert cp.lowers[color] <= proportion  <= cp.uppers[color]
    assert output['objective'] == 1
