import numpy as np
from collections import defaultdict
from .probutil import perturb_2_color
from .utilhelpers import prob_to_det_colors
from ..FairType import FairType


def cluster_color_to_point(pred, color_flag, num_colors):
    """Map a (cluster, color) tuple to a list of points belonging to said
    cluster+color combination.

    Arguments:
    ---
    pred (list[int]): i-th index is the cluster point i belongs to
    color_flag (list[int]): i-th index is the color point i belongs to
    num_colors (int): number of colors
    """
    cluster_color_map = defaultdict(list)
    for i in range(len(color_flag)):
        cluster_color_map[pred[i], color_flag[i]].append(i)
    # for point_ix, (cluster, color) in enumerate(zip(pred, color_flag)):
    #     cluster_color_map[cluster, color].append(point_ix)
    return cluster_color_map


def get_correlated_scramble(color_flag, p_acc, pred, num_colors=2):
    """Flip together points that receive the same color and cluster assignment
    maps tuple[int, int] to list[int]: maps cluster-color tuples to a list
    of points in the given cluster and of said color.
    """
    # NOTE: return type changed from {'var': colors_array} to colors_array, breaking dependent functions
    # if color flag passed in a dictionary mapping vraible of interest to color
    # labels, unpack said dict
    if type(color_flag) == dict:
        assert len(color_flag) == 1
        var = list(color_flag.keys())[0]
        color_flag = color_flag[var]
    if len(color_flag) > len(pred):
        color_flag = color_flag.nonzero()[1]
    cluster_color_map = cluster_color_to_point(pred, color_flag, num_colors)
    # TODO: potentially define prob_vals etc. below this block to ensure the perturbed labels are used when desired
    for cluster, color in cluster_color_map:
        # TODO: implement handling of multicolor case
        # Flip all points: perturb a color with pr. 1 - p_acc
        if np.random.rand() > p_acc:
            # Sample a color and flip all points in the current cluster-color to it
            flipped_color = (color + np.random.randint(1, num_colors)) % num_colors
            for point_ix in cluster_color_map[cluster, color]:
                color_flag[point_ix] = flipped_color
    return color_flag


def flatten_cluster_assignments(assignments, num_clusters):
    """Takes the cluster assignments matrix from output['assignments'] in
    example_2_color.py and flattens it into the format that
    get_fairness_violation() accepts"""
    assignments = np.reshape(assignments, (-1, num_clusters))
    return (assignments >= 1).nonzero()[1]

def scrambled_representations(color_flag, p_acc, pred, assignments, lowers, uppers, m, num_scrambles=15, num_colors=2):
    """Returns the fairness violations after running num_scrambles many
    scrambles (realization of the data).
    As a sanity check, the violation for robust clustering should be no more than the budget
    but for the probabilistic case it might be pretty bad.
    NOTE: if a cluster is empty, its corresponding lb and ub will be 0 and 1

    Arguments
    pred: cluster assignments from vanilla clustering
    assignments: cluster assignments (in matrix form) from _fair_ clustering
    """
    assert len(color_flag.keys()) == 1  # only one variable of interest 
    var = list(color_flag.keys())[0]
    old_color_flag = list(color_flag[var])
    assignments = np.reshape(assignments, (len(pred), -1))
    num_clusters = assignments.shape[1]
    # like pred, but for the actual assignment instead of the vanilla clustering assignment
    # returns the y column values i.e., the i-th entry says which cluster point i was assigned to
    assignment_ix_row, assignment_ix_col = (assignments >= 1).nonzero()
    # Ensure each point is assigned to exactly one cluster
    assert all(assignment_ix_row == np.arange(0, assignments.shape[0]))
    # cluster assignment of each point
    assignment_ix = assignment_ix_col

    assert len(np.unique(pred)) == assignments.shape[1]  # sanity check on the number of clusters/dimensions of assignment
    # list-of-lists for lower and upper bounds; each row corresponds to the cluster bounds of each scramble
    cluster_szs = defaultdict(int)
    for cluster_ix in assignment_ix:
        cluster_szs[cluster_ix] += 1
    # Fairness violations
    lambdas = []
    for _ in range(num_scrambles):
        color_flag = get_correlated_scramble(color_flag, p_acc, pred)
        lambdas.append(get_fairness_violation(color_flag[var], assignment_ix,
                lowers, uppers, cluster_szs, num_clusters, num_colors, m))
    return lambdas


def get_fairness_violation(var_color_flag, assignments, lowers, uppers, cluster_szs, num_clusters, num_colors, m, m_hto, m_toh):
    """Return the fairness violation \lambda given the set of points, labels,
    and cluster assignments; as well as the fairness violations at each cluster.

    Arguments
    ---
    color_flag list(int): color label of each point
    assignments list(int): _fair_ cluster assignment of each point
    lowers list(float): l_h values for each color
    uppers list(float): u_h values for each color
    cluster_szs dict(int, int): map between cluster index and cluster size
    num_clusters (int): number of clusters
    num_colors (int): number of colors
    m (int): perturbation budget in integer number of possible points to perturb

    Returns a tuple of type (float, list[float]) consisting of (maximum
    fractional fairness violation, fractional violation per cluster).
    """
    assert len(assignments) == len(var_color_flag)
    cluster_color_map = cluster_color_to_point(assignments, var_color_flag, num_colors)
    # additive fairness violation lambda
    fairness_vio = 0
    per_cluster_fair_vio = []
    for cluster in range(num_clusters):
        cluster_fair_vio = 0
        cluster_sz = cluster_szs[cluster]
        # If there is an empty cluster, then the fairness violation is the worst case
        if not cluster_sz:
            per_cluster_fair_vio.append(0)
            continue
        for color in range(num_colors):
            num_pts_of_color = len(cluster_color_map[cluster, color])
            num_pts_other_color = cluster_sz - num_pts_of_color
            max_pts_of_color = num_pts_of_color + min(m_toh[color], num_pts_other_color)
            min_pts_of_color = max(0, num_pts_of_color - min(m_hto[color], num_pts_other_color))
            # update with upper and lower bound fractional violation for current color
            upper_vio = max_pts_of_color / cluster_sz - uppers[color]
            lower_vio = lowers[color] - min_pts_of_color / cluster_sz
            cluster_fair_vio = max(cluster_fair_vio, upper_vio, lower_vio)
            fairness_vio = max(fairness_vio, upper_vio, lower_vio)
        per_cluster_fair_vio.append(cluster_fair_vio)
    return fairness_vio, per_cluster_fair_vio


def wrapped_get_fairness_violation(prob_vals, assignments, lowers, uppers, num_clusters, num_points, num_colors, m_frac, m_hto_frac, m_toh_frac, fair_type, num_samples=None):
    """Wrapper for get_fairness_violation; arguments are the same except for

    prob_vals 2D numpy.ndarray: matrix of point-color assignments;
            dimensions: (num_pts x num_clusters)
    assignments list(int): can take output['assignment'] format list of 0-1s
    """
    flattened_assignments = flatten_cluster_assignments(assignments, num_clusters)
    assert len(flattened_assignments) == num_points
    # When computing fairness violation for probabilistic, do the following:
    # 1. switch to matrix to deterministic colors, using the likeliest colors
    # 2. set m to 0, as the violations should come only from the probabilistic
    #    model's label realizations
    if any(0 < p < 1 for p in prob_vals.flatten()):
        prob_vals = prob_to_det_colors(prob_vals)
        p_acc = 1 - m_frac
        m_frac = 0
    # flatten the colors matrix into an array, get true color assignments
    var_color_flag = prob_vals.nonzero()[1]
    # Get a realization for the color_flags given each color is accurate w.p. p_acc
    assert (fair_type != FairType.PROB) or (num_samples is not None), "Probabilistic clustering needs num_samples"
    # Want a single sample for robust and deterministic clustering
    if num_samples is None or fair_type != FairType.PROB:
        num_samples = 1
    fairness_violations = []
    for _ in range(num_samples):
        if fair_type == FairType.PROB:
            var_color_flag_sample = get_correlated_scramble(var_color_flag, p_acc, flattened_assignments)
        else:  # in robust or deterministic clustering, so don't change the color flags
            var_color_flag_sample = var_color_flag
        cluster_szs = defaultdict(int)
        for cluster in flattened_assignments:
            cluster_szs[cluster] += 1
        # cluster_szs_arr = np.array([cluster_szs[i] for i in range(num_clusters)])
        # print('in wrapped_get_fairness_violation: cluster szs: ', cluster_szs_arr)
        m_hto = np.ceil(np.minimum(m_hto_frac, m_frac) * num_points)
        m_toh = np.ceil(np.minimum(m_toh_frac, m_frac) * num_points)
        m = np.ceil(m_frac * num_points)
        fairness_violations.append(
            get_fairness_violation(
                var_color_flag_sample,
                flattened_assignments,
                lowers,
                uppers,
                cluster_szs,
                num_clusters,
                num_colors,
                m,
                m_hto,
                m_toh,
            )
        )
    if len(fairness_violations) == 1:
        return fairness_violations[0]
    # i.e. the fair_type is probabilistic and num_scrambles > 1
    else:
        return fairness_violations

def get_centers(dist, df, r, k):
    """Helper function to robust algorithm for choosing clusters. Exits early if
    more than k centers are used.
    
    Arguments
    ---
    dist: function returning the distance between two points
    df: points to return centers from 
    r: distance parameter r
    k: allowable number of centers

    Returns tuple of (center indexes, list of points selected to be clusters centers)
    """
    num_points = len(df)
    unmarked = set(range(num_points))
    values = df.values
    centers_ixs = []
    while unmarked:
        point_ix = unmarked.pop()
        centers_ixs.append(point_ix)
        if len(centers_ixs) > k:
            break
        to_mark = []
        for other_point_ix in unmarked:
            if dist(values[point_ix], values[other_point_ix]) <= 2 * r:
                to_mark.append(other_point_ix)
        unmarked.difference_update(to_mark)
    return centers_ixs, [df.values[ix] for ix in centers_ixs]


def get_binary_search_params(df, distance, cost_fun_string):
    """Returns hi and smallest_diff binary search parameters. Here is
    smallest_diff corresponds to the binary search's stopping condition of
    hi - lo < smallest_diff.
    
    Params
    ------
    distance:
        distance function taking two vectors

    Returns
    -------
    tuple (hi, smallest_diff)
    """
    values = np.array(df.values)
    values.sort(axis=0)
    # this is the distance between a vector consisting of minimum values at each
    # coord. and another with maximum values at each coord.
    hi = distance(values[0], values[-1])
    def smallest_nonzero_diff(col):
        smallest_diff = float('inf')
        for i in range(len(col)-1):
            diff = col[i+1] - col[i]
            if diff != 0:
                smallest_diff = min(smallest_diff, diff)
        # NOTE: Take sqrt because distance being used is Euclidean
        if cost_fun_string != 'euclidean':
            raise Exception("Adjust the smallest diff if changing the distance")
        return np.sqrt(smallest_diff)
    diffs = np.apply_along_axis(smallest_nonzero_diff, 0, values)
    smallest_diff = np.min(diffs)
    assert smallest_diff != float('inf'), "It appears all points are the same"
    return hi, smallest_diff


def scramble_statistics(fairness_violations, res):
    """Add statistics to dictionary res.
    """
    # lbss = np.asarray(lbss)
    # ubss = np.asarray(ubss)
    # # Add mean
    # res['mean_lb'] = np.mean(lbss, axis=0).tolist()
    # res['mean_ub'] = np.mean(ubss, axis=0).tolist()
    # # Add min lb and max ub
    # res['min_lb'] = np.min(lbss, axis=0).tolist()
    # res['max_ub'] = np.max(ubss, axis=0).tolist()
    # Report the worst case realization 
    res['fairness_violation'] = max(fairness_violations)


def scramble_driver(color_flag, p_acc, pred, assignments, lowers, uppers, output, m):
    """Driver to run scramble tests and cache the results."""
    fairness_violation = scrambled_representations(color_flag, p_acc, pred, assignments, lowers, uppers, m)
    scramble_statistics(fairness_violation, output)

def worst_fair_vio(lowers, uppers):
    worst_fair_vio = 0
    for l, u in zip(lowers, uppers):
        worst_fair_vio = max(1 - u, l, worst_fair_vio)
    return worst_fair_vio
