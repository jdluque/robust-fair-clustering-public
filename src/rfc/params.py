from .FairType import FairType


class RobustParams:
    """Parameters for robust clustering passed to the clustering function. All
    error budget paramters are a fraction corresponding to the portion of the
    dataset with corruptible labels.
    
    Parameters
    ----------
    m_toh: List[float]
        list of m_toh values where i-th entry corresponds to group i
    m: float
        m value for the current clustering experiment
    all_ms: list[float]
        all values of m in the current set of experiments
    """

    def __init__(self, m_toh, m_hto, m, all_ms):
        self.m_toh = m_toh
        self.m_hto = m_hto
        self.m = m
        self.all_ms = all_ms
        self.max_m = max(all_ms)


class ClusteringParams:
    """Parameters passed to the fair clustering function.
    
    test_df: (pd.DataFrame)
        Manually pass in a df for testing purposes.
    test_colors: (np.array[int])
        Manually passed in color labels
    """

    def __init__(
        self,
        dataset,
        config_file,
        data_dir,
        num_clusters,
        deltas,
        max_points,
        L=0,
        p_acc=1.0,
        lowers=[],
        uppers=[],
        fair_type=FairType.PROB,
        robust_params: RobustParams = None,
        seed=24,
        test_df=None,
        test_colors=None,
    ):
        self.dataset = dataset
        self.config_file = config_file
        self.data_dir = data_dir
        self.num_clusters = num_clusters
        self.deltas = deltas
        self.max_points = max_points
        self.L = L
        self.p_acc = p_acc
        self.lowers = lowers
        self.uppers = uppers
        self.fair_type = fair_type
        self.robust_params = robust_params
        self.seed=seed
        self.test_df = test_df
        self.test_colors = test_colors

        if (test_df is None) != (test_colors is None):
            raise ValueError("Must provide both, or no, testing parameters")
        