import numpy as np


def find_min_delta(m, reprs):
    """Returns the minimum valid delta such that the u_h and l_h values do not
    have to be resized by fair_clustering_2_color.py
    
    m: largest error budget (m, m_toh, or m_hto) in consideration
    reprs: list of representation ratio of each color"""
    min_delta = float('-inf')
    num_colors = len(reprs)
    for h in range(num_colors):
        min_u_h = reprs[h] + m
        max_l_h = reprs[h] - m
        if min_u_h >= 1 or max_l_h <= 0:
            return "Not possible without resizing"
        delta_u = 1 - reprs[h] / min_u_h
        delta_l = 1 - max_l_h / reprs[h]
        min_delta = max(min_delta, delta_u, delta_l)

    for h in range(num_colors):
        assert reprs[h] / (1 - min_delta) < 1, "min_delta causes too large u_h"
        assert reprs[h] * (1 - min_delta) > 0, "min_delta causes too small l_h"

    return min_delta

def get_m_vectors_from_pwise_matrix(M, E):
    """Return m_toh and m_hto vectors given aggregate error budget m and
    pairwise errors matrix E.
    """
    M_arr = np.array(M)
    m_toh = np.sum(M_arr, axis=0).tolist()
    m_hto = np.sum(M_arr, axis=1).tolist()
    return m_toh, m_hto

def get_m_vectors_pwise_and_agg(m, m_toh, m_hto):
    """Aggreagte error budget and m_toh, m_hto vectors, return the error vectors
    after taking appropriate minimums with m.
    """
    fixed_m_toh = np.minimum(m, m_toh)
    fixed_m_hto = np.minimum(m, m_hto)
    return fixed_m_toh, fixed_m_hto


def get_valid_lowers_uppers(m, m_toh, m_hto, reprs, delta=None, tol=1E-3):
    """Returns tightest possible lowers, uppers giving a feasible fair robust
    clustering instance.
    
    Parameters
    ---
    m: largest budget error in fraction of points to mess up
    m_toh list[float]: vector of maximum number of false positives for each class h.
        Expressed as a fraction of all points in the dataset
    m_hto list[float]: vector of maximum number of false negatives for each class h.
        Expressed as a fraction of all points in the dataset
    reprs dict[int, float]: representation of each color
    delta: expansion parameter for lower/upper values beyond tightest valid value
    tol: additive growth to avoid numerical errors
    """
    m_toh, m_hto = get_m_vectors_pwise_and_agg(m, m_toh, m_hto)
    lowers, uppers = [], []
    num_colors = len(reprs)
    for h in range(num_colors):
        upper = reprs[h] + m_toh[h] + tol
        lower = reprs[h] - m_hto[h] - tol
        uppers.append(upper)
        lowers.append(lower)
    if delta is not None:
        lowers = [val * (1 - delta) for val in lowers]
        uppers = [val / (1 - delta) for val in uppers]
    assert all(l > 0 for l in lowers), "proportionality demand is vacuous" + str(lowers)
    assert all(u < 1 for u in uppers), "clustering instance is infeasible" + str(uppers)
    return lowers, uppers


if __name__ == '__main__':
    print(find_min_delta(.15, [0.6186684361866843, 0.3813315638133156]))