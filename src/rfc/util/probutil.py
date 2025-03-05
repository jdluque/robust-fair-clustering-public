import numpy as np 
import random 

#np.random.seed(0)


def perturb_2_color(color_val,p_acc):
	if color_val==1:
		return p_acc
	else:
		return (1-p_acc)


def form_class_prob_vector(p_acc,num_colors):
    p= [0]*num_colors
    p[0] = p_acc 
    p = [((1-p[0])/(num_colors-1)) if i!=0 else p_ for i,p_ in enumerate(p)]
    return p 

def perturb_memmership(color_val,num_colors,p):
	perturbation = np.random.choice(num_colors,p=p)
	color_val_perturb = (color_val + perturbation) % num_colors
	return color_val_perturb


def sample_colors(color_flag,num_colors,p_acc):
	p = form_class_prob_vector(p_acc,num_colors)
	color_flag_perturb =[perturb_memmership(color_val,num_colors,p) for color_val in color_flag] 
	return color_flag_perturb


def sample_colors_ml_model(prob_vecs,num_colors):
	n = prob_vecs.shape[0] 
	color_flag = n*[0] 
	for idx in range(n):
		color_flag[idx] = np.random.choice(num_colors,p=prob_vecs[idx,:])

	return color_flag


# n is the number of points 
def create_prob_vecs(n,p_acc,num_colors,color_flag):

	probs = (((1-p_acc)/(num_colors-1)))*np.ones((n,num_colors))

	for i in range(n):
		probs[i,color_flag[i]] = p_acc 

	return probs

