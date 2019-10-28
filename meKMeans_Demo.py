## General imports
import numpy as np
import pandas as pd
import os,inspect

# Get this current script file's directory:
loc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Set working directory
os.chdir(loc)
# from myFunctions import gen_FTN_data
# from meSAX import *

# from dtw_featurespace import *
# from dtw import dtw
# from fastdtw import fastdtw

# to avoid tk crash
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## my colors

color_list = ['gold', 'darkcyan','slateblue', 'hotpink', 'indigo', 'firebrick', 'skyblue', 'coral', 'sandybrown', 'mediumpurple',  'forestgreen', 'magenta', 'seagreen', 'greenyellow', 'roaylblue', 'gray', 'lightseagreen']

# Matplotlib default color cycler
# matplotlib.rcParams['axes.prop_cycle']
default_color_list = []
for obj in matplotlib.rcParams['axes.prop_cycle']:
    default_color_list.append(obj['color'])
# combine the two color lists
my_colors =  default_color_list
[my_colors.append(c) for c in color_list]

# my_colors = np.array(['#1f77b4','#2ca02c','#ff7f0e'])


my_colors = np.array(my_colors)

## generate data

# set random seed
np.random.seed(0)

# my sample data
x_mean1 = 0
y_mean1 = 0

x_mean2 = 45
y_mean2 = 13

x_mean3 = 7
y_mean3 = 40

N1 = 100
N2 = 100
N3 = 100

# coords1 = np.random.uniform(0,12,(N1,2))
# coords2 = np.random.uniform(0,5,(N2,2))
coords1 = np.random.randn(N1,2) * 16
coords2 = np.random.randn(N2,2) * 4
coords3 = np.random.randn(N3,2) * 1
outliers = np.array([15,15,23,12]).reshape(2,2)
coords = np.empty((N1+N2+N3+outliers.shape[0],2))


coords[:N1] =  coords1 + (x_mean1,y_mean1)
coords[N1:(N1+N2)] =  coords2 + (x_mean2,y_mean2)
coords[(N1+N2):-2] =  coords3 + (x_mean3,y_mean3)
coords[-2:] = outliers

# # sklearn example data
# n_points_per_cluster = 250
# 
# C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
# C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
# C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
# C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
# C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
# C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
# coords = np.vstack((C1, C2, C3, C4, C5, C6))

## k-means full demo


from meKMeans import DataPoint,KMeans

km = KMeans(coords,k=4,tol=1e-5,max_iter=100)

km.cluster()
labels = km.labels_()
centers = km.centers_()


# plot
plt.figure()
plt.scatter(coords[:,0],coords[:,1],color=my_colors[labels],alpha=0.5)
plt.scatter(centers[:,0],centers[:,1],color=my_colors[np.arange(len(centers))],marker='x')
plt.show()


## pure functions


X = coords
k = 5
max_iter = 5
centers = []
# initialize data points
DPs = [] # list of data points
for i,datapoint in enumerate(X):
    p = DataPoint(i)
    p.coord = X[i] # if not useDistMeasure: p.coord = X[i]
    DPs.append(p)

centers = setSeeds(DPs,centers)

DPs,centers = cluster(DPs,centers)
labels = labels_(DPs)


## plot steps


from meKMeans import DataPoint,KMeans

km = KMeans(coords,k=4,tol=1e-5,max_iter=100)

# initialize manually
km.setSeeds()


# Run this section for the steps: ==============================================
# cluster for one iteration
km.cluster_one_step()
labels = km.labels_()
centers = km.centers_()


# plot
plt.figure()
plt.scatter(coords[:,0],coords[:,1],color=my_colors[labels],alpha=0.5)
# plt.scatter(centers[:,0],centers[:,1],color=my_colors[np.arange(len(centers))],marker='x')
plt.scatter(centers[:,0],centers[:,1],color='red',marker='x')
plt.show()
# ==============================================================================

























