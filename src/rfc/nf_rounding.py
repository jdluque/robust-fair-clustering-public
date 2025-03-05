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


def set_child_node_name(j,i):
    return 'c'+str(j)+'_'+str(i)

def get_center_number(child_node_name):
    return int(child_node_name.split('_')[1]) 

def dot(K, L):
   if len(K) != len(L):
           return 0

   return sum(i[0] * i[1] for i in zip(K, L))


# NOTE: this assumes that x is a 2d numpy array  
def find_proprtions_two_color(x,num_colors,color_prob,num_clusters):
    proportions = np.zeros(num_clusters)

    for cluster in range(num_clusters):
        #print(x[:,cluster])
        #print(color_prob)
        proportions[cluster] = dot(x[:,cluster],color_prob)

    div_total = np.sum(x,axis=0)
    div_total[np.where(div_total == 0)]=1 
    proportions_normalized = proportions/div_total

    return proportions_normalized, proportions






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
                    valid= False 
                else:
                    row_count+=1 
            # if not almost 1 and not almost 0 
            elif abs(x[i,j]) > epsilon: 
                valid= False 
            # if almost 0 
            elif abs(x[i,j]) <= epsilon: 
                x[i,j]=0 

        if row_count ==0:
            valid= False

    return valid , x


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

def min_cost_rounding_2_color(df, clustering_method, centers, distance, color_prob, num_colors, res):


    # number of clusters 
    num_clusters = len(centers)

    # LP fractional assignments in x 
    x = res["partial_assignment"]
    x = np.reshape(x, (-1,num_clusters))
    lp_correct , _ = vet_x(x,epsilon)

    # NOTE: sometimes CPLEX makes mistakes 
    #if not lp_correct:
    #	raise ValueError('Error: LP has negative values.')

    # LP objective 
    lp_sol_val = res["partial_objective"]


    # number of points 
    n = len(df)
    # distance converted to matrix form and is rounded to integer values 
    d = np.round_(np.reshape(scale_up_factor*distance, (-1,num_clusters)))

    print('\n\n') 
    print('NF Rounding ...')
    ( _ , color_prob), = color_prob.items() # color_prob now a list 


    # create DiGraph object 
    G = nx.DiGraph()

    # Step 1: Add all of the node with demand value of -1 
    demand_color_point = [None]*n
    for i in range(n):
        demand_color_point[i] = {'demand':-1, 'color': color_prob[i]} 

    nodes_point = list(range(n)) 
    nodes_attrs_point = zip(nodes_point, demand_color_point)

    G.add_nodes_from(nodes_attrs_point)


    # Step 2: For cluster i, add ceiling(cluster i size) many vertices each with demand 0 and no color 
    assignment_cluster = np.sum(x,axis=0)
    assignment_cluster_floor = np.floor(assignment_cluster) 
    assignment_cluster_ceil = np.ceil(assignment_cluster) 

    for cluster in range(num_clusters):
        for j in range(int(assignment_cluster_ceil[cluster])):
            child_node_name =  set_child_node_name(j,i)
            G.add_node(child_node_name, demand=0, color='NO_COLOR') 




    # Step 3: Connect vertices to the child vertices of each center to which their assigned. See the paper for details on 
    # the connection 
    #print('\n------------\nConnect Node to Child Centers')
    for cluster in range(num_clusters):
        # get the assigned vertices 
        assigned_vertex_indices = np.where(x[:,cluster]>0)[0]
        # get the probabilities of the assigned vertices 
        color_prob_assigned = np.asarray([color_prob[idx] for idx in assigned_vertex_indices]) 
        # sort the indices of the assigned vertices by descending probability value 
        assigned_vertex_indices = assigned_vertex_indices[np.argsort(-color_prob_assigned)]
        # get the corresponding LP assignments for each vertex 
        assigned_vertex_lp = x[assigned_vertex_indices,cluster]

        num_child_vertices = int(assignment_cluster_ceil[cluster])
        num_assigned_vertices = assigned_vertex_indices.shape[0] 

        # acc will store the accumilated probability value 
        acc = np.zeros(num_child_vertices)



        # at_vertex is vertex within assigned_vertex_indices
        at_vertex = 0 
        
        for j in range(num_child_vertices):	
            child_node_name = set_child_node_name(j,cluster)
            acc[j] += assigned_vertex_lp[at_vertex]
            G.add_edge(assigned_vertex_indices[at_vertex],child_node_name,capacity=1,weight=d[assigned_vertex_indices[at_vertex]][cluster])
            # below for debug 
            #G.add_edge(assigned_vertex_indices[at_vertex],child_node_name,capacity=1,weight=assigned_vertex_lp[at_vertex])
            assigned_vertex_lp[at_vertex] -= assigned_vertex_lp[at_vertex] 
            if assigned_vertex_lp[at_vertex]==0:
                at_vertex +=1 
            while acc[j] < 1 and at_vertex<=num_assigned_vertices-1:	
                value_to_add = min(1-acc[j],assigned_vertex_lp[at_vertex])
                acc[j] += value_to_add
                G.add_edge(assigned_vertex_indices[at_vertex],child_node_name,capacity=1,weight=d[assigned_vertex_indices[at_vertex]][cluster])
                # below for debug 
                #G.add_edge(assigned_vertex_indices[at_vertex],child_node_name,capacity=1,weight=value_to_add)
                assigned_vertex_lp[at_vertex] -= value_to_add
                if assigned_vertex_lp[at_vertex]==0:
                    at_vertex +=1 

                
    # Step 4: Add a node for each center 
    for cluster in range(num_clusters):
        node_name = 's'+str(cluster)
        G.add_node(node_name, demand=int(assignment_cluster_floor[cluster]),color='NO_COLOR')



    # Step 5: Add edges between the child centers and their parent center 
    for cluster in range(num_clusters):
        num_child_vertices = int(assignment_cluster_ceil[cluster])
        center_node_name = 's'+str(cluster) 
        for j in range(num_child_vertices):	
            child_node_name = set_child_node_name(j,cluster)
            G.add_edge(child_node_name,center_node_name,capacity=1,weight=0)


    # Step 6: Add a vertex for t 
    demand_t = n- np.sum(assignment_cluster_floor) 
    G.add_node('t',demand=demand_t)

    # Step 7: Add edges between centers and t 
    for cluster in range(num_clusters):
        center_node  = 's'+str(cluster)
        if assignment_cluster[cluster] > assignment_cluster_floor[cluster]:
              G.add_edge(center_node,'t',capacity=1,weight=0)


    # Step 8: Solve the min-cost flow problem 
    flowCost, flowDict = nx.network_simplex(G) 


    # convert solution to assignments x 
    x_rounded = np.zeros((n,num_clusters))

    for node, node_flows in flowDict.items(): 
        if type(node) is int: 
            for child_center, flow in node_flows.items(): 
                if flow==1:
                    x_rounded[node,get_center_number(child_center)]=1



    success_flag , x_rounded = check_rounding_and_clip(x_rounded,epsilon)
    
    if success_flag: 
        print('\nNetwork Flow Rounding Done.\n')
    else: 
        raise ValueError('Error: NF rounding has returned non-integer solution.')


    # Get color proportions for each color and cluster
    rounded_proportions_normalized, rounded_proportions = find_proprtions_two_color(x_rounded,num_colors,color_prob,num_clusters)
    lp_proportions_normalized, lp_proportions = find_proprtions_two_color(x,num_colors,color_prob,num_clusters)
 

    # calculate the objective value according to this
    x_rounded = x_rounded.ravel().tolist()

    if clustering_method == "kmeans" or clustering_method == "kmedian":
        final_cost = dot(x_rounded,distance) 
    else:
        final_cost = max(distance)

    rounded_sol_val = final_cost
    res["objective"] = final_cost
    res['assignment'] = x_rounded 

    res['partial_proportions'] = lp_proportions.ravel().tolist()
    res['proportions'] = rounded_proportions.ravel().tolist()

    res['partial_proportions_normalized'] = lp_proportions_normalized.ravel().tolist()
    res['proportions_normalized'] = rounded_proportions_normalized.ravel().tolist()

    ratio_rounded_lp = rounded_sol_val/lp_sol_val 

    if (ratio_rounded_lp-epsilon)>1:
        raise ValueError('Error: NF rounding has higher cost. Try increasing scale_up_factor.') 
    else:
        pass 
        #print('\n---------\nratio= rounded_sol_val / lp_sol_val = %f' %  ratio_rounded_lp )


    return res 



