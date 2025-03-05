import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from cplex import Cplex
import cplex
import time


from .nf_rounding import min_cost_rounding_2_color 
from .nf_rounding_multi_color import min_cost_rounding_multi_color
from .FairType import FairType
from .util.robustutil import get_centers, get_binary_search_params
from .util.utilhelpers import readd_variables


def fair_partial_assignment(df, centers, uppers, lowers, color_flag, clustering_method, num_colors, L=0, fair_type=FairType.PROB, robust_params=None):
    # NOTE: Legacy code from probabilistic fair clustering. Robust clustering does not have guarantees for these objectives.
    if clustering_method == "kmeans" or clustering_method == "kmedian":
        assert fair_type != FairType.ROBUST, "Robust clustering implemented only for kcenter objective"
        cost_fun_string = 'euclidean' if clustering_method == "kmedian" else 'sqeuclidean'
        new_problem, objective = fair_partial_assignment_lp_solver(df, centers, color_flag, uppers, lowers, cost_fun_string, L, fair_type=fair_type, robust_params=robust_params)
        # Step 5. call the solver

        t1 = time.monotonic()
        new_problem.solve()
        t2 = time.monotonic()
        print("LP solving time = {}".format(t2-t1))

        # problem.solution is a weakly referenced object, so we must save its data
        #   in a dictionary so we can write it to a file later.
        res = {
            "partial_status": new_problem.solution.get_status(),
            "partial_success": new_problem.solution.get_status_string(),
            "partial_objective": new_problem.solution.get_objective_value(),
            "partial_assignment": new_problem.solution.get_values(),
        }

        print('lp solver result: ', res['partial_status'], res['partial_success'])
        # NOTE: here is where the rounding is done
        # comment the first line and uncomment the second to disable the rounnding
        if fair_type in [FairType.DET, FairType.ROBUST]:
            print('running multi_col!!')
            final_res = min_cost_rounding_multi_color(df, clustering_method, centers, objective, color_flag, num_colors, res)
        else:
            final_res = min_cost_rounding_2_color(df, clustering_method, centers, objective, color_flag, num_colors, res) 

        # TODO: Need to grab the objective from here
        rounded_sol_val = final_res["objective"]

        final_res["partial_assignment"] = res["partial_assignment"]
        final_res["partial_objective"] = res["partial_objective"]

        if clustering_method == "kmeans":
            final_res["partial_objective"] = np.sqrt(final_res["partial_objective"])
            final_res["objective"] = np.sqrt(final_res["objective"])

        return final_res

    elif clustering_method == "kcenter":
        cost_fun_string = 'euclidean'
        def distance(x, y) -> float:
            """Returns the distance between vectors x and y."""
            return cdist([x], [y], cost_fun_string).item()
        if fair_type in [FairType.PROB, FairType.DET]:
            problem, objective = fair_partial_assignment_lp_solver(df, centers, color_flag, uppers, lowers, cost_fun_string ,L, fair_type=fair_type, robust_params=robust_params)

            cost_ub = max(objective) + 1
            cost_lb = 0
            least_feasible_cost = cost_ub
            cheapest_feasible_lp = problem
            cheapest_feasible_obj = objective

            # TODO: Might want to make this condition more precise than adding 0.1
            while cost_ub > cost_lb + 0.1:
                cost_mid = (cost_ub + cost_lb)/2.0
                new_problem, objective = fair_partial_assignment_lp_solver(df, centers, color_flag, uppers, lowers, cost_fun_string, L, fair_type=fair_type, robust_params=robust_params)
                to_delete = [idx for idx, el in enumerate(objective) if el > cost_mid]
                if len(to_delete) > 0:
                    new_problem.variables.delete(to_delete)

                new_problem.solve()
                new_stats = new_problem.solution.get_status()
                if new_stats == 1: # optimal!
                    cost_ub = cost_mid
                    least_feasible_cost = cost_mid
                    cheapest_feasible_lp = new_problem
                    cheapest_feasible_obj = objective

                elif new_stats == 3: #infeasible
                    cost_lb = cost_mid

                else:
                    raise ValueError("LP solver stat code {}".format(new_stats) + " with cost {}".format(cost_mid))

            # Make "objective" and "assignment" arrays that match the original number of variables
            num_centers = len(centers)
            nr_variables = len(objective)
            num_points = len(df)
            assignment, objective = readd_variables(cheapest_feasible_lp, problem, cheapest_feasible_obj, num_centers)

            # problem.solution is a weakly referenced object, so we must save its data
            #   in a dictionary so we can write it to a file later.
            res = {
                "partial_status" : cheapest_feasible_lp.solution.get_status(),
                "partial_success" : cheapest_feasible_lp.solution.get_status_string(),
                # NOTE: the objective we care about is least_feasible_cost, which is captured below
                "partial_objective" : cheapest_feasible_lp.solution.get_objective_value(),
                "partial_assignment" : assignment,
            }

            # Flatten color flags
            # TODO: Consider streamlining the code for prob and not prob. E.g.,
            # we could use the color flag and just multiply by p_acc when
            # building the LP. This avoids building the n x h matrix of colors
            # for each point
            flattened_color_flag = {}
            if fair_type == FairType.PROB:
                for var in color_flag:
                    flattened_color_flag[var] = np.argmax(color_flag[var], axis=1).tolist()
            else:
                flattened_color_flag = color_flag
            final_res = min_cost_rounding_multi_color(df, clustering_method,centers, objective, flattened_color_flag, num_colors, res) 

        else:  # robust fair clustering
            print('Solving here')
            hi, smallest_diff = get_binary_search_params(df, distance, cost_fun_string)
            lo = 0
            num_points = len(df)
            # max allowable number of centers
            k = len(centers)
            # replace centers with something that makes sense: since we care only
            # about using the problem for variable names, we can pass in the
            # first k points as centers
            # NOTE: Any point can be a center so full problem should have every point as a center
            problem, objective = fair_partial_assignment_lp_solver(df, df.values[:k], color_flag, uppers, lowers, cost_fun_string,L, fair_type=fair_type, robust_params=robust_params)

            res, final_res = {}, {}
            least_feasible_cost = float('inf')
            finished = False
            while not finished:
                # This is the last iteration
                if 3 * (hi - lo) < smallest_diff:
                    R = hi
                    finished = True
                else:
                    R = (lo + hi) / 2
                center_ixs, centers = get_centers(distance, df, R, k)
                if len(centers) <= k:
                    # Delete x_j_i if dist from pt j to center i exceeds 3*R
                    to_delete = []
                    values = df.values
                    for j in range(num_points):
                        # lp variables are written as x_j_lp_ix
                        # correspondence between lp_ix and cluster_ix
                        for lp_ix, i in enumerate(center_ixs):
                            if distance(values[i], values[j]) > 3 * R:
                                to_delete.append(f'x_{j}_{lp_ix}')
                    new_problem, new_objective = fair_partial_assignment_lp_solver(df, centers, color_flag, uppers, lowers, cost_fun_string,L, fair_type=fair_type, robust_params=robust_params)
                    new_problem.variables.delete(to_delete)
                    new_problem.solve()
                    status = new_problem.solution.get_status()
                    print('Solution status: ', status)
                    if status == 1:  # optimal
                        print('====================Found feasible sol')
                        hi = R
                        # Readd variables
                        assignment, objective = readd_variables(new_problem, problem, new_objective, k)
                        res = {
                            "partial_status" : new_problem.solution.get_status(),
                            "partial_success" : new_problem.solution.get_status_string(),
                            # NOTE: we care about the max distance, not this
                            "partial_objective" : new_problem.solution.get_objective_value(),
                            "partial_assignment" : assignment,
                        }
                        # TODO: Update least feasible cost and other info only if it beats previous best
                        # the assignments are to one of the k previously selected centers
                        assignment_matrix = np.array(assignment).reshape((num_points, k))
                        # selected_centers[i] = j says the i-th point has a center of point j; both zero-indexed
                        selected_centers = np.argmax(assignment_matrix, axis=1)
                        selected_centers = [df.values[center_ixs[i]].tolist() for i in selected_centers]
                        # NOTE: centers just needs to be as long as the number of possible centers, which = k
                        new_final_res = min_cost_rounding_multi_color(df, clustering_method, [0]*k, objective, color_flag, num_colors, res) 
                        # Manually compute distances between points and their assigned centers; then take the max
                        max_dist = 0
                        for i in range(len(df)):
                            dist_to_center = distance(
                                df.values[i], selected_centers[i]
                            )
                            max_dist = max(max_dist, dist_to_center)
                        if max_dist < least_feasible_cost:
                            least_feasible_cost = max_dist
                            final_res = new_final_res
                    else:  # infeasible
                        print('====================Infeasible LP')
                        lo = R
                else:
                    lo = R

        print(res)
        final_res["partial_assignment"] = res["partial_assignment"]
        final_res["partial_objective"] = res["partial_objective"]

        rounded_cost = 0
        for idx, value in enumerate(final_res["assignment"]):
            rounded_cost = max(rounded_cost, value * objective[idx])

        if clustering_method == 'kcenter':
            # NOTE: max point-assigned center distance: i.e., k-center objective
            final_res["objective"] = least_feasible_cost
        else:
            final_res["objective"] = np.sqrt(rounded_cost)
        final_res["partial_objective"] = np.sqrt(least_feasible_cost)
        final_res["partial_assignment"] = assignment

        return final_res

    else:
        print("Not a valid clustering method. Available methods are: " 
              "\'kmeans\', \'kmedian\', and \'kcenter\'.")
        return None


'''
The main function in this file is fair_partial_assignment_lp_solver.
This function takes as input a collection of data points, a list of 
cluster centers, a list of colors of each points, and fairness parameters.
It then constructs the fair assignment lp and solves it. It returns 
a fractional assignment of each point to a cluster center.  

Input Description:
    df: a dataframe of the input points
    centers: a list of the euclidean centers found via clustering
    color_flag : a list of color values for all the points -- helpful in adding constraints to the lp solver
    alpha: dict where the keys are colors and the values is the alpha for that color
    beta: dict where the keys are colors and the values are the beta for that color

Output Description:
    res: a dictionary with the following keys:
        "status": an integer code depicting the outcome of the lp solver
        "success": a string to interpret the above code
        "objective": the objective function value
        "assignment": the assignment of values to all the LP variables
        
Reading the "assignment" array:
    the variables are created in the following order:
        for all j in points
            for all i in centers 
                create x_{j}_{i} 
    So the assignment array, which is a list of floats, corresponds to this variable order
'''


def fair_partial_assignment_lp_solver(df, centers, color_flag, uppers, lowers, cost_fun_string, L, fair_type=FairType.PROB, robust_params=None):
    # There are primarily five steps:
    # 1. Initiate a model for cplex
    # 2. Declare if it is minimization or maximization problem
    # 3. Add variables to the model. The variables are generally named.
    #    The upper bounds and lower bounds on the range for the variables
    #    are also mentioned at this stage. The coefficient of the objective
    #    functions are also entered at this step
    # 4. Add the constraints to the model. The constraint matrix, denoted by A,
    #    can be added in three ways - row wise, column wise or non-zero entry wise.
    # 5. Finally, call the solver.

    # Step 1. Initiate a model for cplex.

    print("Initializing Cplex model")
    problem = Cplex()

    # Step 2. Declare that this is a minimization problem

    problem.objective.set_sense(problem.objective.sense.minimize)

    # Step 3.   Declare and  add variables to the model. The function
    #           prepare_to_add_variables (points, center) prepares all the
    #           required information for this stage.
    #
    #    objective: a list of coefficients (float) in the linear objective function
    #    lower bounds: a list of floats containing the lower bounds for each variable
    #    upper bounds: a list of floats containing the upper bounds for each variable
    #    variable_name: a list of strings that contains the name of the variables

    print("Starting to add variables...")
    t1 = time.monotonic()

    # NOTE: lower_bounds=0, upper_bounds = 1, variable_names = {x_j_i}, objective=list of corresponding coefficients for each x_i_j 
    objective, lower_bounds, upper_bounds, variable_names = prepare_to_add_variables(df, centers, cost_fun_string, fair_type=fair_type)
    assert len(variable_names) == len(set(variable_names))
    problem.variables.add(obj=objective,
                        lb=lower_bounds,
                        ub=upper_bounds,
                        names=variable_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding variable = {}".format(t2-t1))

    # Step 4.   Declare and add constraints to the model.
    #           There are few ways of adding constraints: rwo wise, col wise and non-zero entry wise.
    #           Assume the constraint matrix is A. We add the constraints row wise.
    #           The function prepare_to_add_constraints_by_entry(points,center,colors,alpha,beta)
    #           prepares the required data for this step.
    #
    #  constraints_row: Encoding of each row of the constraint matrix
    #  senses: a list of strings that identifies whether the corresponding constraint is
    #          an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
    #  rhs: a list of floats corresponding to the rhs of the constraints.
    #  constraint_names: a list of string corresponding to the name of the constraint

    # TODO: Modify added constraints from probabilistic to the desired deterministic w/ error budget
    print("Starting to add constraints...")
    t1 = time.monotonic()
    objects_returned = prepare_to_add_constraints(df, centers, color_flag, lowers, uppers, L, fair_type, robust_params)
    constraints_row, senses, rhs, constraint_names = objects_returned
    assert len(constraint_names) == len(set(constraint_names))
    problem.linear_constraints.add(lin_expr=constraints_row,
                                   senses=senses,
                                   rhs=rhs,
                                   names=constraint_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding constraints = {}".format(t2-t1))

    # Optional: We can set various parameters to optimize the performance of the lp solver
    # As an example, the following sets barrier method as the lp solving method
    # The other available methods are: auto, primal, dual, sifting, concurrent

    #problem.parameters.lpmethod.set(problem.parameters.lpmethod.values.barrier)

    return problem, objective

# This function creates necessary stuff to prepare for variable
# addition to the cplex lp solver
# Input: See the input description of the main function
# Output:
#   objective: a list of coefficients (float) in the linear objective function
#   lower bounds: a list of floats containing the lower bounds for each variable
#   upper bounds: a list of floats containing the upper bounds for each variable
#   variable_name: a list of strings that contains the name of the variables
# The ordering of variables are as follows: for every point, we create #centers many
# variables and put them together before moving onto the next point.
# for j in range(num_points)
#   for i in range(num_centers)
#       create x_j_i


def prepare_to_add_variables(df, centers, cost_fun_string, fair_type=FairType.PROB):
    num_points = len(df)
    num_centers = len(centers)

    # NOTE: this reverses the order we are used to x_i_j, this is x_j_i 
    # Name the variables -- x_j_i is set to 1 if j th pt is assigned to ith center
    variable_names = []
    # Add x_j_i variables and their respecitve ubs and lbs
    variable_names.extend(["x_{}_{}".format(j,i) for j in range(num_points) for i in range(num_centers)])
    num_x_vars = num_points * num_centers
    lower_bounds = [0] * num_x_vars
    upper_bounds = [1] * num_x_vars
    # Cost function: Minimize the weighted sum of the distance from each point
    #   to each center.
    objective = cost_function(df, centers, cost_fun_string)
    # Add other variables required for the robust LP
    # NOTE: new robust alg does not need any new variables
    # if fair_type == FairType.ROBUST:
    #     # Add the p and tilde-p dual variables
    #     # NOTE: Handling two color case only 
    #     num_colors = 2
    #     variable_names.extend([f'{sym}p_{i}_{j}^{h}'
    #                             for sym in ['~', '']
    #                             for j in range(num_points)
    #                             for i in range(num_centers)
    #                             for h in range(num_colors)])  
    #     num_p_vars = 2 * num_colors * num_points * num_centers
    #     lower_bounds.extend([0] * num_p_vars)
    #     # TODO: Potentially look at it
    #     upper_bounds.extend([cplex.infinity] * num_p_vars)
    #     # Add z_ih's
    #     variable_names.extend([f'{sym}z_{i}_{h}'
    #                             for sym in ['~', '']
    #                             for i in range(num_centers)
    #                             for h in range(num_colors)])  
    #     num_z_vars = 2 * num_colors * num_centers
    #     lower_bounds.extend([0] * num_z_vars)
    #     upper_bounds.extend([cplex.infinity] * num_z_vars)
    #     # Need to extend objective with the values for the other variables
    #     objective += ([0] * (len(variable_names) - num_x_vars))

    # All values should be between 0 and 1 -- if a single tuple is provided,
    #   then it will be applied to all points, according to scipy docs.
    # TODO: Can replace the bloc above with total_variables = len(variable_names)
    
    # NOTE: below is always true 
    # assert len(variable_names) == total_variables


    return objective, lower_bounds, upper_bounds, variable_names

# Cost function: Minimize the weighted sum of the distance from each point
#   to each center.
# Implementation details:
#   cdist(XA, XB, metric='euclidean', *args, **kwargs): Compute distance between each pair of the two
#   collections of inputs. metric = 'sqeuclidean' computes the squared Euclidean distance. cdist returns a
#   |XA| by |XB| distance matrix. For each i and j, the metric dist(u=XA[i], v=XB[j]) is computed and stored in the
#   (i,j)th entry.
#   ravel(a,order='C'): A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
# The ordering of the final output array is consistent with the order of the variables.
# Note that all_pair_distance is an array and we need to convert it to a list before returning


def cost_function(df,centers, cost_fun_string):
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    return all_pair_distance.ravel().tolist()

# This function prepares for constraints addition by non zero entries
# Input: See the input description of the main function
# Output:
#  constraints_row: Encoding of each row of the constraint matrix
#  senses: a list of strings that identifies whether the corresponding constraint is
#          an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
#  rhs: a list of floats corresponding to the rhs of the constraints.
#  constraint_names: a list of string corresponding to the name of the constraint
#
# There are two types of constraints.
# 1. Assignment constraints and 2. Fairness constraints
#     1. first we have no_of_points many assignment constraints (sum_{i} x_j_i = 1 for all j)
#     2. then we have num_center*num_colors*2 many fairness constraints.
#        fairness constraints are indexed by center, and then color. We first have all beta constraints, followed
#        by alpha constraints


# NOTE: L is used as a lower bound for variables?
def prepare_to_add_constraints(df, centers, color_flag, lowers, uppers, L, fair_type=FairType.PROB, robust_params=None):
    num_points = len(df)
    num_centers = len(centers)
    num_colors = len(lowers)

    # The following steps constructs the assignment constraint. Each corresponding row in A
    # has #centers many non-zero entries.
    constraints_row, rhs = constraint_sums_to_one(num_points, num_centers)
    sum_const_len = len(rhs)
    senses = ["E" for _ in range(sum_const_len)]

    # Grab the variable of interest
    assert len(color_flag) == 1
    var = list(color_flag.keys())[0]
    var_color_flag = color_flag[var]
    # var_beta, var_alpha = beta[var], alpha[var]

    # The following steps constructs the fairness constraints. There are #centers * # colors * 2
    # many such constraints. Each of them has #points many non-zero entries.
    # Fairness constraints added here
    if fair_type in [FairType.PROB, FairType.DET]:
        # Probabilistic or Prob should be the same, but the color flags should
        # already adjusted accordingly
        color_constraint, color_rhs = constraint_two_color(num_points, num_centers, var_color_flag, lowers, uppers)
        constraints_row.extend(color_constraint)
        rhs.extend(color_rhs)

        # NOTE: This is to enforce the lower bound
        # If L=0, then no lower bound is enforced 
        constraints_LB, rhs_LB = constraint_lower_bound_cluster(num_points, num_centers, L)
        constraints_row.extend(constraints_LB)
        rhs.extend(rhs_LB)

        # The assignment constraints are of equality type and the rest are less than equal to type
        # NOTE: no need to add anything for lower bounded cluster size 
        senses.extend(["L" for _ in range(len(rhs) - sum_const_len)])

    elif fair_type == FairType.ROBUST:
        # Group assignment matrix A, robustness parameter T, and noise budget matrix M
        # A, T, M  = robust_params['A'], robust_params['T'], robust_params['M'] 
        all_constraints, all_rhs, all_senses = robust_constraints(num_points, num_centers, var_color_flag, lowers, uppers, robust_params, num_colors)
        constraints_row.extend(all_constraints)
        rhs.extend(all_rhs)
        senses.extend(all_senses)

    # Name the constraints
    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]
    return constraints_row, senses, rhs, constraint_names

def robust_constraints(num_points, num_centers, var_color_flag, lowers, uppers, robust_params, num_colors=2):
    m_to_h = robust_params.m_toh
    m_h_to = robust_params.m_hto
    # print('lowers ', lowers)
    # print('uppers ', uppers)
    def A(j, h):
        # Like the 0-1 matrix A, returns 1 if point j has color h, else 0
        return int(var_color_flag[j] == h)
    assert all(u > l for u, l in zip(uppers, lowers))
    all_constraints, all_rhs, all_senses = robust_constraints_upper(num_points, num_centers, A, uppers, num_colors, m_to_h)
    incoming_constraints, incoming_rhs, incoming_senses = robust_constraints_lower(num_points, num_centers, A, lowers, num_colors, m_h_to)
    all_constraints += incoming_constraints
    all_rhs += incoming_rhs
    all_senses  += incoming_senses 
    # incoming_constraints, incoming_rhs, incoming_senses = robust_constraints_dual(num_points, num_centers, A, num_colors)
    # all_constraints += incoming_constraints
    # all_rhs += incoming_rhs
    # all_senses  += incoming_senses 
    return all_constraints, all_rhs, all_senses

def robust_constraints_dual(num_points, num_centers, A, num_colors):
    """Deprecated by updated robust clustering algorithm.
    
    Add constraints with dual variables(27e-f)."""
    all_constraints, all_rhs, all_senses = [], [], [] 
    for sym in ['', '~']:
        # Add constraints, one constraint per cluster center and group
        variables, weights, rhs, senses = defaultdict(list), defaultdict(list), {}, {}
        for i in range(num_centers):
            for j in range(num_points):
                for h in range(num_colors):
                    constr_tup = (sym, i, j, h)
                    variables[constr_tup].append(f'{sym}z_{i}_{h}')
                    weights[constr_tup].append(1)
                    variables[constr_tup].append(f'{sym}p_{i}_{j}^{h}')
                    weights[constr_tup].append(1)
                    # Move rhs to left and flip signs
                    variables[constr_tup].append(f'x_{j}_{i}')
                    if sym == '~':
                        weights[constr_tup].append(-A(j, h))
                    else:
                        weights[constr_tup].append(A(j, h) - 1)
                    rhs[constr_tup] = 0
                    senses[constr_tup] = "G"
        for i in range(num_centers):
            for j in range(num_points):
                for h in range(num_colors):
                    constr_tup = (sym, i, j, h)
                    all_constraints.append([variables[constr_tup], weights[constr_tup]])
                    all_rhs.append(rhs[constr_tup])
                    all_senses.append(senses[constr_tup])
    return all_constraints, all_rhs, all_senses


def robust_constraints_lower(num_points, num_centers, A, lowers, num_colors, m_h_to):
    # Add constraints, one constraint per cluster center and group
    variables, weights, rhs, senses = defaultdict(list), defaultdict(list), {}, {}
    # u_h constraints
    for i in range(num_centers):
        for h in range(num_colors):
            for j in range(num_points):
                variables[i, h].append(f'x_{j}_{i}')
                weights[i, h].append(-lowers[h] + A(j, h))
            rhs[i, h] = m_h_to[h]
            senses[i, h] = "G"

            # # - sum_{j in P} x_{i,j} A_{j,h}
            # for j in range(num_points):
            #     variables[i, h].append(f'x_{j}_{i}')
            #     # Add the term from moving rhs to lhs
            #     weights[i, h].append(-1 * A(j, h) + lowers[h] - T)
            # # m_{h to} ~z_{i,h}
            # variables[i, h].append(f'~z_{i}_{h}')
            # # m_h_to = 1
            # weights[i, h].append(m_h_to[h])
            # # sum_{j in [n]} p_{i,j}^h
            # for j in range(num_points):
            #     variables[i, h].append(f'~p_{i}_{j}^{h}')
            #     weights[i, h].append(1)
            # # move the rhs to the left
            # # Originally: ... <= (T - l_h) * sum_{j in P} x_i,j
            # # for j in range(num_points):
            # #     variables[i, h].append(f'x_{j}_{i}')
            # #     weights[i, h].append(var_alpha - T)
            # rhs[i, h] = 0
            # senses[i, h] = "L"
    all_constraints, all_rhs, all_senses = [], [], [] 
    for i in range(num_centers):
        for h in range(num_colors):
            all_constraints.append([variables[i, h], weights[i, h]])
            all_rhs.append(rhs[i, h])
            all_senses.append(senses[i, h])
    return all_constraints, all_rhs, all_senses


def robust_constraints_upper(num_points, num_centers, A, uppers, num_colors, m_to_h):
    # Add constraints, one constraint per cluster center and group
    variables, weights, rhs, senses = defaultdict(list), defaultdict(list), {}, {}
    # (27c) u_h constraints
    # forall i in S, h in [\ell],
    for i in range(num_centers):
        for h in range(num_colors):
            for j in range(num_points):
                variables[i, h].append(f'x_{j}_{i}')
                weights[i, h].append(-uppers[h] + A(j, h))
            rhs[i, h] = -m_to_h[h]
            senses[i, h] = "L"

            # sum_{j in P} x_{i,j} A_{j,h}
            # for j in range(num_points):
            #     variables[i, h].append(f'x_{j}_{i}')
            #     # This includes subtracting a term due to moving the rhs to lhs
            #     # move the rhs to the left
            #     # Originally: ... <= (uh + T) * sum_{j in P} x_i, j
            #     weights[i, h].append(A(j, h) - uppers[h] - T)
            # # m_{to h} z_{i,h}
            # variables[i, h].append(f'z_{i}_{h}')
            # # TODO: fill in m_to_h val
            # weights[i, h].append(m_to_h[h])
            # # sum_{j in [n]} p_{i,j}^h
            # for j in range(num_points):
            #     variables[i, h].append(f'p_{i}_{j}^{h}')
            #     weights[i, h].append(1)
            # rhs[i, h] = 0
            # senses[i, h] = "L"
    all_constraints, all_rhs, all_senses = [], [], [] 
    for i in range(num_centers):
        for h in range(num_colors):
            # TODO: Catch repeated variables here
            # if not len(variables[i, h]) == len(set(variables[i, h])):
            #     print(sorted(variables[i, h]))
            all_constraints.append([variables[i, h], weights[i, h]])
            all_rhs.append(rhs[i, h])
            all_senses.append(senses[i, h])
    return all_constraints, all_rhs, all_senses


# this function adds the constraint that every client must be assigned to exactly one center
# Implementation:
# Example: assume 3 points and 2 centers. Total of 6 variables: x_0_0,x_0_1 for first point, and so on.
#  row 0         x00 + x01                 = 1
#  row 1                  x10 + x11        = 1
#  row 2                         x20 + x21 = 1
# The constraints are entered in the following format:
# [[['x_0_0', 'x_0_1'], [1, 1]],
#  [['x_1_0', 'x_1_1'], [1, 1]],
#  [['x_2_0', 'x_2_1'], [1, 1]]]


def constraint_sums_to_one(num_points, num_centers):
    constraints = [[["x_{}_{}".format(j, i) for i in range(num_centers)], [1] * num_centers] for j in range(num_points)]
    rhs = [1] * num_points
 
    return constraints, rhs


# this function adds the fairness constraint
# Implementation:
# The following example demonstrates this
# Example:  Assume 3 points and 2 centers and 2 colors.
#           Total of 6 variables: x00,x01 for first point, x10,x11 for second point and so on.
#           Assume 1 and 2 belong to color 0. Let a == alpha and b == beta
#
#     (b1-1) x00             + (b1-1) x10            + b1 x20                 <= 0    center 0, color 1, beta
#        b2  x00             +   b2   x10            + (b2-1) x20             <= 0    center 0, color 2, beta
#                 (b1-1) x01           + (b1-1) x11              + b1 x21     <= 0    center 1, color 1, beta
#                    b2  x01            +  b2   x11              + (b2-1) x21 <= 0    center 1, color 2, beta
#
#     (1-a1) x00            + (1-a1) x20             - a1 x30                 <= 0    center 1, color 1, alpha
#       - a2 x00             - a2 x20                + (1-a2) x30             <= 0    center 1, color 2, alpha
#               (1-a1) x10             + (1-a1) x21              - a1 x31     <= 0    center 2, color 1, alpha
#              - a2 x10                 - a2   x21              + (1-a2) x31 <= 0    center 2, color 2, alpha
#
# Below we depict the details of the entries (the first 4 rows)
# [
# [['x_0_0','x_1_0','x_2_0'],[b1-1,b1-1,b1]]
# [['x_0_0','x_1_0','x_2_0'],[b2,b2,b2-1]]
# [['x_0_1','x_1_1','x_2_1'],[b1-1,b1-1,b1]]
# [['x_0_1','x_1_1','x_2_1'],[b2,b2,b2-1]]
# ...]
def constraint_two_color(num_points, num_centers, prob_vals, lowers, uppers):
    # beta_val and alpha_val should depend on the color of point j
    num_colors = len(lowers)
    # infer the fair type based on whether prob_vals is all integers
    if all(float(p).is_integer() for p in np.asarray(prob_vals).ravel()):
        fair_type = FairType.DET
    else:
        fair_type = FairType.PROB
    if fair_type == FairType.DET:
        # lowers[prob_vals[j]] is l_{h_\ell} where h_\ell is the group that point j belongs to 
        # TODO: Double check whether below should use lowers[h] or lowers[prob_vals[[j]]]
        beta_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                            [(lowers[h] - int(prob_vals[j] == h)) for j in range(num_points)]]
                            for i in range(num_centers) for h in range(num_colors)]
        alpha_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                            [(int(prob_vals[j] == h) - uppers[h]) for j in range(num_points)]]
                            for i in range(num_centers) for h in range(num_colors)]
    else:  # for FairType.PROB, need constraints for each color
        beta_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                            [lowers[h] - prob_vals[j][h] for j in range(num_points)]]
                            for i in range(num_centers) for h in range(num_colors)]
        alpha_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                            [prob_vals[j][h] - uppers[h] for j in range(num_points)]]
                            for i in range(num_centers) for h in range(num_colors)]

    constraints = beta_constraints + alpha_constraints
    number_of_constraints = len(constraints)
    rhs = [0] * number_of_constraints
    return constraints, rhs

def constraint_color(num_points, num_centers, color_flag, beta, alpha):

    beta_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                         [beta[color] - 1 if color_flag[j] == color else beta[color] for j in range(num_points)]]
                        for i in range(num_centers) for color, _ in beta.items()]


    alpha_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                          [np.round(1 - alpha[color], decimals=3) if color_flag[j] == color else (-1) * alpha[color]
                           for j in range(num_points)]]
                         for i in range(num_centers) for color, _ in beta.items()]
    constraints = beta_constraints + alpha_constraints
    number_of_constraints = num_centers * len(beta) * 2
    rhs = [0] * number_of_constraints
    return constraints, rhs


# NOTE: This forces the minimum number of points assigned to a cluster to be >=L
def constraint_lower_bound_cluster(num_points, num_centers, L):

    constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)], [-1] *  num_points] for i in range(num_centers)]
    rhs = [-L]*num_centers
 
    return constraints, rhs
