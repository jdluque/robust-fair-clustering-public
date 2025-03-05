"""Generate the .sh files for running experiments.

Ex. usage: `python write_experiments_config.py > target.sh`
"""

# dataset names and their respective colors
per_dataset_num_color = {
    "adult_sex": 2,
    "bank_binary_marital": 2,
    "census1990_ss_age_3_classes": 3,
}
# one of det, prob, or robust
num_clusters = 10
max_points = 32561
joint_exps = True
deltas = "0.1"
fair_types = ["DET", "PROB", "ROBUST"]
# small_m budgets
# varied_budgets = "np.linspace(.001, .010, 10)"
# large_m budgets
varied_budgets = "np.linspace(.01, .10, 10)"
# Uesd to determine the folder to save cvs in
exps_family = "large_m"
# used to compute fairness violations of probabilistic clustering
num_samples = 200

default_line = """
[DEFAULT]
config_file = config/dataset_configs.ini
max_points = {max_points}
data_dir = output/
num_clusters = {num_clusters}
budgets = {budgets}
joint_exps = {joint_exps}
deltas = {deltas}
exps_family = {exps_family}
num_samples = {num_samples}
"""
output = """
[{dataset}_{fair_type}]
dataset = {dataset}
exp_name = {exp_name}
fair_type = FairType.{fair_type}
num_colors = {num_colors}
"""
config_file_content = ""
exp_names = []
for fair_type in fair_types:
    for dataset, num_colors in per_dataset_num_color.items():
        budgets = varied_budgets
        if num_colors > 2 and fair_type == "PROB":
            continue
        exp_name = fair_type
        exp_names.append(dataset + "_" + exp_name)
        # Do not run probabilistic baseline on multiple colors
        config_file_content += output.format(
            dataset=dataset,
            fair_type=fair_type,
            exp_name=exp_name,
            num_colors=num_colors,
        )
        default_line = default_line.format(
            exp_name=exp_name,
            deltas=deltas,
            max_points=max_points,
            num_clusters=num_clusters,
            budgets=budgets,
            exps_family=exps_family,
            joint_exps=joint_exps,
            num_samples=num_samples,
        )
config_file_content = default_line + config_file_content
# print(config_file_content)
executable_name = f"{exps_family}_experiments"
config_file = f"config/{executable_name}.ini"
bash_file = f"src/{executable_name}.sh"
bash_scipt = ""
for exp_name in exp_names:
    bash_scipt += f"python src/example_vary_budget.py {exp_name} {config_file}\n"
with open(config_file, "w") as f:
    f.write(config_file_content)
with open(bash_file, "w") as f:
    f.write(bash_scipt)
# print(bash_scipt)
