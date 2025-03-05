import numpy as np
import networkx as nx
import math
from scipy.spatial.distance import cdist
from cplex import Cplex
import time
import matplotlib.pyplot as plt


# epsilon is used for clipping 
epsilon = 0.001
scale_up_factor = 1000

# NOTE: this assumes that x is a 2d numpy array  
def find_proprtions(x,num_colors,color_flag,num_clusters):
	proportions = np.zeros((num_colors,num_clusters))
	for color in range(num_colors):
		rel_color_indices = [i for i, x in enumerate(color_flag) if x == color]
		color_sum = np.sum(x[rel_color_indices,:],axis=0)
		for cluster in range(num_clusters):
			proportions[color,cluster] = color_sum[cluster]
	div_total = np.sum(x,axis=0)
	div_total[np.where(div_total == 0)]=1 
	proportions_normalized = proportions/div_total
	return proportions_normalized, proportions

def check_rounding_and_clip(x,epsilon):
	n,m = x.shape
	valid = True 
	for i in range(n):
		row_count = 0 
		for j in range(m):
			# if almost 1 
			if abs(x[i,j]-1)<= epsilon:
				x[i,j] = 1 
				if row_count==1:
					print('fail')
					print(x[i,j])
					valid= False 
				else:
					row_count+=1 
			# if not almost 1 and not almost 0 
			elif abs(x[i,j]) > epsilon: 
				print('fail')
				print(x[i,j])
				valid= False 
			# if almost 0 
			elif abs(x[i,j]) <= epsilon: 
				x[i,j]=0 

		if row_count ==0:
			print('fail')
			print(x[i,j])
			valid= False

	return valid , x


def dot(K, L):
   if len(K) != len(L):
   		return 0

   return sum(i[0] * i[1] for i in zip(K, L))

def vet_x(x,epsilon):
	n,m = x.shape
	valid = True 
	for i in range(n):
		row_count = 0 
		for j in range(m):
			# if almost 1 
			if (x[i,j]+epsilon)<0:
				valid= False
				#print(x[i,j])

	return valid , x


def min_cost_rounding_multi_color(df, clustering_method, centers, distance, color_flag, num_colors, res):
	# number of clusters 
	num_clusters = len(centers)
	num_points = len(df)
	# 
	lp_sol_val = res["partial_objective"]
	# LP fractional assignments in x 
	x = res["partial_assignment"]
	# Clip solution so that only x variables are sent (i.e., mno other variables from the robust LP)
	x = x[:(num_clusters * num_points)]
	x = np.reshape(x, (-1,num_clusters))
	distance = distance[:(num_clusters * num_points)]

	lp_correct , _ = vet_x(x,epsilon)


	#if not lp_correct:
	#	raise ValueError('Error: LP has negative values.')

	# number of points 
	n = len(df)
	# distance converted to matrix form and is rounded to integer values 
	d = np.round_(np.reshape(scale_up_factor*distance, (-1,num_clusters)))
	# get the color_flag in list form 
	print('NF Rounding ...')

	( _ , color_flag), = color_flag.items()

	# Define a graph 
	G = nx.DiGraph()

	# Step 1: Add graph vertices with the right demand and color values  
	demand_color_point = [None]*n
	for i in range(n):
		demand_color_point[i] = {'demand':-1, 'color': color_flag[i]} 

	nodes_point = list(range(n)) 
	nodes_attrs_point = zip(nodes_point, demand_color_point)

	# Step 1 DONE 
	G.add_nodes_from(nodes_attrs_point)

	# Step 2: Add colored centers with the right demand and color values
	for color in range(num_colors):

		rel_color_indices = [i for i, x in enumerate(color_flag) if x == color]

		# demand: is an array where the ith index is for the ith cluster  
		demand = np.floor(np.sum(x[rel_color_indices,:],axis=0)) 
		

		for cluster in range(num_clusters): 
			node_name = 'c'+str(color)+str(cluster)
			G.add_node(node_name, demand=demand[cluster],color=color) 



	# Step 3: Add edges between points and colored centers with the right cost and capacity 
	for color in range(num_colors):
		# get the points with the right color 
		rel_color_indices = [i for i, x in enumerate(color_flag) if x == color]

		for cluster in range(num_clusters): 
			# get the points assigned to this cluster 
			rel_cluster_indices = np.where(x[:,cluster]>0)[0].tolist() 

			assigned_points = list(set(rel_color_indices) & set(rel_cluster_indices)) 

			colored_center = 'c'+str(color)+str(cluster)

			edges = [(i,colored_center,{'capacity':1,'weight':d[i,cluster]}) for i in assigned_points] 

			G.add_edges_from(edges) 


	# Step 4: Add centers (the set S)
	assignment_cluster = np.sum(x,axis=0)
	assignment_cluster_floor = np.floor(assignment_cluster) 

	for cluster in range(num_clusters):
		demands_colors = 0 
		for color in range(num_colors):
			node_name = 'c'+str(color)+str(cluster)
			demands_colors += G.nodes[node_name]['demand']

		demand_cluster_nf = assignment_cluster_floor[cluster]-demands_colors

		node_name = 's'+str(cluster)
		G.add_node(node_name, demand=demand_cluster_nf) 


	# Step 5: Add edges between colored centers and centers 
	for color in range(num_colors):
		rel_color_indices = [i for i, x in enumerate(color_flag) if x == color]

		# assignment is an array where the ith index is for the ith cluster
		assignment = np.sum(x[rel_color_indices,:],axis=0) 
		assignment_floor = np.floor(assignment)

		for cluster in range(num_clusters):
			center_node  = 's'+str(cluster)
			if assignment[cluster] > assignment_floor[cluster]:
				colored_center = 'c'+str(color)+str(cluster)
				G.add_edge(colored_center,center_node,capacity=1,weight=0)
	

	# Step 6: Add t 
	demand_t = n- np.sum(assignment_cluster_floor)
	G.add_node('t',demand=demand_t)

	# Step 7: Add edges between the centers and t 
	for cluster in range(num_clusters):
		
		center_node  = 's'+str(cluster)
		if assignment_cluster[cluster] > assignment_cluster_floor[cluster]:
		   G.add_edge(center_node,'t',capacity=1,weight=0)
	


	# Step 7: Solve the network flow problem 
	flowCost, flowDict = nx.network_simplex(G) 


	# Step 8: convert solution to assignments x 
	x_rounded = np.zeros((n,num_clusters))

	for node, node_flows in flowDict.items(): 
		if type(node) is int: 
			for center, flow in node_flows.items(): 
				if flow==1:
					string_to_remove = 'c'+str(color_flag[node]) 
					x_rounded[node,int(center.replace(string_to_remove,''))]=1


	success_flag , x_rounded = check_rounding_and_clip(x_rounded,epsilon)
	
	if success_flag: 
		print('\nNetwork Flow Rounding Done.\n')
	else: 
		raise ValueError('NF rounding has returned non-integer solution.')

	# Get color proportions for each color and cluster
	lp_proportions_normalized, lp_proportions = find_proprtions(x,num_colors,color_flag,num_clusters)
	rounded_proportions_normalized, rounded_proportions = find_proprtions(x_rounded,num_colors,color_flag,num_clusters)

	# calculate the objective value according to this
	x_rounded = x_rounded.ravel().tolist()

	#assert clustering_method == "kcenter"
	if clustering_method == "kmeans" or clustering_method == "kmedian":
		final_cost = dot(x_rounded,distance) 
	# TODO: Update this for k-center alg
	else:
		# this for the k-center 
		distances_to_center = [a*b for a,b in zip(x_rounded,distance)] 
		final_cost = max(distances_to_center)

	res["objective"] = final_cost
	res['assignment'] = x_rounded 
	rounded_sol_val = final_cost


	res['partial_proportions'] = lp_proportions.ravel().tolist()
	res['proportions'] = rounded_proportions.ravel().tolist()

	res['partial_proportions_normalized'] = lp_proportions_normalized.ravel().tolist()
	res['proportions_normalized'] = rounded_proportions_normalized.ravel().tolist()


	if lp_sol_val:
		ratio_rounded_lp = rounded_sol_val/lp_sol_val 
	else:
		ratio_rounded_lp = float('inf')




	distances_to_center = [a*b for a,b in zip(x_rounded,distance)] 
	kcenter_radius = max(distances_to_center) 



	# NOTE: removed because the LPs objective value is not the largest radius as it should have been
	# if (ratio_rounded_lp-epsilon)>1:
	# 	raise ValueError('NF rounding has higher cost. Try increasing scale_up_factor.') 
	# else:
	# 	pass 
		#print('\n---------\nratio= rounded_sol_val / lp_sol_val = %f' %  ratio_rounded_lp )



	return res 