import numpy as np 

epsilon = 0.0001 

def dot(K, L):
   if len(K) != len(L):
   		return 0

   return sum(i[0] * i[1] for i in zip(K, L))

# find the proportions based on given assignments 
def find_proprtions_two_color(x,num_colors,color_prob,num_clusters):
	x = np.reshape(x , (-1,num_clusters)) 
	proportions = np.zeros(num_clusters)

	for cluster in range(num_clusters):
		print('dot...: ', dot(x[:,cluster],color_prob))
		proportions[cluster] = dot(x[:,cluster],color_prob)

	div_total = np.sum(x,axis=0)
	div_total[np.where(div_total == 0)]=1 
	proportions_normalized = proportions/div_total

	return proportions_normalized, proportions, np.sum(x,axis=0)


# find the proportions based on given assignments, multi-color 
def find_proprtions_multi_color(x,num_colors,prob_vecs,num_clusters):
	x = np.reshape(x , (-1,num_clusters)) 
	div_total = np.sum(x,axis=0)
	div_total[np.where(div_total == 0)]=1 

	proportions = np.zeros((num_clusters,num_colors)) 
	proportions_normalized = np.zeros((num_clusters,num_colors)) 
	for cluster in range(num_clusters):
		for color in range(num_colors):
			proportions[cluster,color] = np.dot(x[:,cluster],prob_vecs[:,color])

		proportions_normalized[cluster,:] = proportions[cluster,:]/div_total[cluster]

	return proportions_normalized, proportions, np.sum(x,axis=0)

# find  maxViol_from_proprtion 
def maxViol_from_proprtion(alpha, beta, num_clusters, proportions, sizes):
	gamma_fair =0 

	for counter in range(num_clusters):
		upper_viol = proportions[counter]-alpha*sizes[counter]
		lower_viol = beta*sizes[counter]-proportions[counter]
		max_viol = max(upper_viol,lower_viol)
		if max_viol>gamma_fair:
			gamma_fair = max_viol

	return gamma_fair 


# find  maxViol_from_proprtion, multi color. alpha and beta are arrays 
def maxViol_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes):
	gamma_fair =0 

	for col in range(num_colors):
		for cluster_idx in range(num_clusters):
			upper_viol = proportions[cluster_idx,col]-alpha[col]*sizes[cluster_idx]
			lower_viol = beta[col]*sizes[cluster_idx]-proportions[cluster_idx,col]

			max_viol = max(upper_viol,lower_viol)
			if max_viol>gamma_fair:
				gamma_fair = max_viol

	return gamma_fair 


# find  maxViol_from_proprtion, multi color. alpha and beta are arrays 
def maxViol_Normalized_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes):
	gamma_normalized = 0 

	for col in range(num_colors):
		for cluster_idx in range(num_clusters):
			upper_viol = proportions[cluster_idx,col]-alpha[col]*sizes[cluster_idx]
			lower_viol = beta[col]*sizes[cluster_idx]-proportions[cluster_idx,col]

			max_viol_norm = max(upper_viol,lower_viol)/sizes[cluster_idx] 

			if max_viol_norm>gamma_normalized:
				gamma_normalized = max_viol_norm

	return gamma_normalized 

# find  maxViol_from_proprtion, multi color. alpha and beta are arrays 
def maxRatioViol_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes):
	gamma_fair =0 


	for col__ in range(num_colors):
		for cluster_index in range(num_clusters):
			if sizes[cluster_index] != 0 :
				upper_viol = proportions[cluster_index,col__]-alpha[col__]*sizes[cluster_index]
				lower_viol = beta[col__]*sizes[cluster_index]-proportions[cluster_index,col__]
				max_viol = max(upper_viol,lower_viol)
				denominator = 1*sizes[cluster_index]
				max_viol_prop = max_viol/denominator
	
				if max_viol_prop>gamma_fair:
					gamma_fair = max_viol_prop

	return gamma_fair 


# find max_Viol multi_color 
def max_RatioViol_multi_color(x,num_colors,prob_vecs,num_clusters,alpha,beta):
	_ , proportions, sizes = find_proprtions_multi_color(x,num_colors,prob_vecs,num_clusters)
	return maxRatioViol_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes)




# find max_Viol
def max_Viol(x,num_colors,color_prob,num_clusters,alpha,beta):
	_ , proportions, sizes = find_proprtions_two_color(x,num_colors,color_prob,num_clusters)
	return maxViol_from_proprtion(alpha, beta, num_clusters, proportions, sizes)


# find max_Viol multi_color 
def max_Viol_multi_color(x,num_colors,prob_vecs,num_clusters,alpha,beta):
	_ , proportions, sizes = find_proprtions_multi_color(x,num_colors,prob_vecs,num_clusters)
	return maxViol_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes)

# 
def max_Viol_Normalized_multi_color(x,num_colors,prob_vecs,num_clusters,alpha,beta):
	_ , proportions, sizes = find_proprtions_multi_color(x,num_colors,prob_vecs,num_clusters)
	return maxViol_Normalized_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes)





def find_balance(x,num_colors, num_clusters,color_prob,proportion_data_set):
	proportions_normalized, _ , sizes = find_proprtions_two_color(x,num_colors,color_prob,num_clusters)
	min_balance = 1 
	balance_unfair = np.zeros(num_clusters)

	x = np.reshape(x , (-1,num_clusters)) 

	for i in range(num_clusters):
		if sizes[i]!=0:
			balance = min(proportions_normalized[i]/proportion_data_set , proportion_data_set/proportions_normalized[i])
			if (proportions_normalized[i]==0) or (proportion_data_set==0):
				pass 
		else: 
			balance = 10 

		balance_unfair[i] = balance
		if min_balance>balance:
			min_balance = balance 

	return min_balance




# find balance multi color, proportion_data_set is an array 
def find_balance_multi_color(x,num_colors, num_clusters,color_prob,proportion_data_set):
	proportions_normalized, _ , sizes = find_proprtions_multi_color(x,num_colors,color_prob,num_clusters)
	min_balance = 1 
	balance_unfair = np.zeros(num_clusters)

	for clust in range(num_clusters):
		for col_ in range(num_colors):
			if sizes[clust]!=0:
				balance = min(proportions_normalized[clust,col_]/proportion_data_set[col_] , proportion_data_set[col_]/proportions_normalized[clust,col_])
			else: 
				balance = 10 

			balance_unfair[clust] = balance
			if min_balance>balance:
				min_balance = balance 

	return min_balance





# for assignment for color-blind 
def x_for_colorBlind(preds,num_clusters):
	x = np.zeros((len(preds),num_clusters)) 
	for idx,p in enumerate(preds): 
		x[idx,p] = 1 

	return x.ravel().tolist() 

def prob_vec_two_to_multi_color(prob_vec):
	"""Take a prob vec from 2 color form to multi-color form.
	i.e., a 2-color prob_vec looks like [.2, .8, .2, .2, ....]
	with entries corresponding to membership in class 1. Turn this into a matrix
	where col 0 is 1-p and col 1 is p, for each entry p in the 2d prob_vec.
	"""
	# var_name, prob_vec = prob_vec.items()
	probs_matrix = np.zeros((len(prob_vec), 2))
	for i, p in enumerate(prob_vec):
		probs_matrix[i, 0] = 1 - p
		probs_matrix[i, 1] = p
	return probs_matrix

def prob_to_det_colors(prob_vals):
	"""prob_vals is a colors matrix with probability values for color assignments.
	Returns a 1-D array corresponding to the likeliest color for each point."""
	colors = [np.argmax(row) for row in prob_vals]
	rv = np.zeros_like(prob_vals)
	for row, color in zip(rv, colors):
		row[color] = 1
	return rv

def readd_variables(new_problem, original_problem, new_objective, num_centers):
	"""Re-adds deleted variables to a cplex.problem.
	
	Arguments
	---
	original_problem is cplex.problem with variable names intact
	num_centers should be the num of centers in the original problem, which
			could be inferred.
	"""
	num_variables = len(original_problem.variables.get_names())
	assignment = [0] * num_variables
	objective = [0] * num_variables
	for new_idx, var_name in enumerate(new_problem.variables.get_names()):
		parts = var_name.split('_')
		j = int(parts[1]) # point number
		i = int(parts[2]) # center number
		old_idx = j*num_centers + i
		# old_idx = problem.variables.get_index(var_name)

		old_name = original_problem.variables.get_names(old_idx)
		if old_name != var_name:
			raise Exception("Old name: {} and var_name: {} do not match for new_idx = {} and old_idx = {}".format(old_name, var_name, new_idx, old_idx))
		objective[old_idx] = new_objective[new_idx]
		assignment[old_idx] = new_problem.solution.get_values(new_idx)
	return assignment, objective
