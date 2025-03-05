# README


## Requirements

`Python3.6` is expected for this program because currently the CPLEX solver being used expects that version of Python.

To install the non-CPLEX dependencies, use `pip install -r requirements.txt`.

To install CPLEX, visit the IBM website and navigate to the proper license (student, academic, professional, etc.), and follow the installation guide provided by IBM.

You can install the `rfc` package we provide using `pip install .` from the project's root directory.

## Recreating plots

You can run the experiments by running the `.sh` files under `src/`.
E.g., Running `bash src/large_m_experiments.sh` will recreate the csv files used for the plots of m vs objective and fairness violations in the main paper.
Similarly, running `bash src/noise_models_comparison.sh` will create the csvs for the plots comparing different settings of $m^+$ and $m^-$.


The required csv's are already in the project so running the notebook `notebooks/evaluation.ipynb` suffices to recreate plots.

## Running Example Script

If dependencies are installed you should be able to run the examples:

Further the following examples, will vary a parameter: number of clusters for the first two and the lower bound on the cluster size for the last. They produce a csv file under the Results folder:
* `example_vary_budget.py`: Will vary the noise parameter m.
* `example_noise_models.py`: Varies the per-class (color) noise.
* `example_vary_max_points.py`: Varies the number of points n; useful for observing changes in running time.
* `example_vary_budget_new_det.py`: Evaluates the deterministic algorithm more forgivingly by relaxing the proportionality constraints as the noise parameter m grows.

## Running your Own Tests

To run one of your own tests, edit the following three things:

1. Create your own config file in `config/` e.g., copy `small_m_experiments.ini`.
2. Create your own `.sh` file such as in `src/` or directly run `python your_exp_name config/your_config.ini` where `your_exp_name` is a header in "[]" `config/your_config.ini`.

The `dataset` string in `config/your_config.ini` reflects the dataset to be used and should come from `config/dataset_configs.ini`. You can create new datasets by adding a new entry to this file.

## Description of Output Format

The output from a trial will be a new file for each run with the timestamp: `%Y-%m-%d-%H:%M:%S`. A run is defined as a combination of `num_cluster` and `delta` in the config file. For example, if two values for `num_clusters` and two deltas are specified, then 4 runs will occur.

Each output file is in JSON format, and can be loaded using the `json` package from the standard library. The data is held as a dictionary format and can be accessed by using string key names of the following fields: 
* `num_clusters` : The number of clusters used for this trial.
* `alpha` : Dictionary holding the alphas for various colors. First key is the attribute (ie. sex), and second key is the color within that attribute (ie. male).
* `beta` : Dictionary holding the betas for various colors.
First key is the attribute (ie. sex), and second key is the color within that attribute (ie. male).
* `unfair_score` : Clustering objective score returned by vanilla clustering. 0 if `violating` is True.
* `objective` : Clustering objective returned by the fair clustering algorithm.
* `sizes` : List holding the sizes of the clusters returned by vanilla clustering. Empty list if `violating` is True.
* `attributes` : Dictionary holding the points that belong to each color group. First key is the attribute that is being considered (ie. sex), second key is the color group within that attribute that the point belongs to (ie. male).
* `centers` : List of centers found by vanilla clustering. Empty list if `violating` is True.
* `points` : List of points used for fair clustering or violating LP. Useful if the dataset has been subsampled to know which points were chosen by the subsampling method.
* `assignment`: List (sparse) of points and their assigned cluster. There are (# of points) * (# of centers) entries in assignments. For each point `i`, we say that it is assigned to that cluster `f` if `assignment[i*(# of centers) + f] == 1`.
* `name` : String name of the dataset chosen. Will use name from `dataset_configs.ini` file.
* `clustering_method` : String name of the clustering method used.
* `scaling` : Boolean of whether or not data was scaled.
* `delta` : delta value used for this run. Note that this is not the overlap but rather the variable involved in the reparameterization of alpha and beta. beta = (1 - delta) and alpha = 1 / (1 - delta).
* `time` : Float that is the time taken for the LP to be solved.
* `cluster_time` : Float that is the time taken for the vanilla clustering to occur. 0 if `violating` is true.

### Other fields in CSVs

CSVs are found under `experiments/experiment_name` where `experiment_name` is given by the `exps_family` field in the config being run (e.g., `config/your_config.ini`).

* `budgetViolation` : The greatest _fractional_ violation among all clusters.
* `perClusterVio` : The fractional violation achieved at each cluster.
* `worstPossibleFairVio` : The maximum achievable fractional fairness violation, given the proportionality parameters.
* `cluster_szs` : List of the number of points at each cluster.
* `additive_violations` : List of additive violations at each cluster. Compared to fractional violations, this is the raw number of points exceeding or undershooting the violated proportionality bound.
* `max_add_violation` : The greatest additive violation i.e. a max over `additive_violations`.

## References 

This code is built on top of the code for "Fair Algorithms for Clustering":  https://github.com/nicolasjulioflores/fair_algorithms_for_clustering
