# # Get this current script file's directory:
# import os,inspect
# loc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# # Set working directory
# os.chdir(loc)


import numpy as np
from scipy.spatial import distance
import copy


## class
class DataPoint:
    def __init__(self,index,isCenter=False):
        self.coord = None # coordinates, could be None if only given a distance matrix
        self.index = index # the order numbering from given data
        self.label = None # cluster assignment
        self.isCenter = isCenter # a boolean that indicates if this data point is a cluster center
        
    
class KMeans:
    def __init__(self,X,k,dist_func=distance.minkowski,useDistMeasure=False,initSeeds=None,tol=1e-5,max_iter=100):
        '''
        Inputs:
            X: Data matrix, np.array(n_samples*n_dimensions)
            k: number of clusters, integer
            dist_func: distance measure function, default is distance.minkowski
            useDistMeasure: Boolean variable, default is False.
                            If set to True, then X should be a distance matrix, np.array(n_samples * n_samples)
                            Also, the k-medoids would be calculated instead of k-means
                            Default distance measure is Euclidean distance
            initSeeds: assigning the initial cluster centers np.array(k * n_dimensions)
                       if set to None, then random generates centers
            tol: tolerance for convergence, if centers difference < tol, then is considered as converged
            max_iter: max iterations, integer
        Outputs:
        
        Attributes:
            labels: the cluster labels for each data point. (0 to k-1)
        
        '''
        # initialize
        self.X = X
        self.k = int(k)
        self.dist = dist_func
        self.useDistMeasure = useDistMeasure
        self.initSeeds = initSeeds
        self.max_iter = max_iter
        self.labels = np.zeros((X.shape[0],))
        self.centers = []
        self.tol = tol
        
        if not initSeeds==None and k != initSeeds.shape[0]:
            print('number of clusters does not match with initSeeds!')
        elif self.k < 2:
            print('should have at least 2 clusters!')
        
        # initialize data points
        DPs = [] # list of data points
        for i,datapoint in enumerate(X):
            p = DataPoint(i)
            p.coord = X[i] # if not useDistMeasure: p.coord = X[i]
            DPs.append(p)
        self.DPs = DPs
    
    
    def setSeeds(self):
        '''
        Set up initial centers
        '''
        # initialize
        DPs = self.DPs
        k = self.k
        initSeeds = self.initSeeds
        
        centers = []
        for i in range(k):
            c = DataPoint(index=i,isCenter=True)
            c.label = int(i)
            centers.append(c)
        
        if initSeeds == None: # generate starting points
            N = len(DPs) # number of data points
            indices = np.random.randint(10,size = k)
            for i in range(k):
                centers[i].coord = DPs[indices[i]].coord
            
        else: # use initSeeds
            for i in range(k):
                centers[i].coord = initSeeds[i]
        
        self.centers = centers
        
    
    def cluster_one_step(self):
        '''
        Runs only one step
        ----------
        '''
        # # initialize centers
        # self.setSeeds()
        
        # one iteration:
        self.assign_labels()
        self.update_centers()
    
    
    def cluster(self):
        '''
        Runs the algorithm
        ----------
        '''
        # initialize
        max_iter = self.max_iter
        
        # initialize centers
        self.setSeeds()
        
        # start iterations:
        for i in range(max_iter):
            pre_centers = copy.deepcopy(self.centers) # save centers of previous step
            self.assign_labels()
            self.update_centers()
            # if converged, then stop
            if self.converged(pre_centers):
                print('converged at {} iteration'.format(i+1))
                break
            
    
    def converged(self,pre_centers):
        '''
        check if converged
        ----------
        Inputs:
            pre_centers: centers of previous step
        Outputs:
            returns True if converged
        '''
        # initialize
        centers = self.centers
        tol = self.tol
        dist = self.dist
        N = len(centers)
        # calculate center changes
        sum = 0
        for i,c in enumerate(centers):
            pc = pre_centers[i]
            d = dist(c.coord,pc.coord)
            sum += d
        
        if sum <= tol:
            return(True)
        else:
            return(False)
    
        
        
        
            
    
    def assign_labels(self):
        '''
        Assignment step
        '''
        # initialize
        DPs = self.DPs
        centers = self.centers
        dist = self.dist
        
        for p in DPs:
            dists = [] # list of distances to centers
            for c in centers:
                d = dist(p.coord,c.coord)
                dists.append(d)
            dists = np.array(dists)
            index = np.argmin(dists) # find the closest center
            p.label = centers[index].label # assign cluster label
                
        self.DPs = DPs # update
    
    
    def update_centers(self):
        '''
        Update step
        '''
        # initialize
        DPs = self.DPs
        centers = self.centers
        # dist = self.dist
        clusters = {} # dictionary of lists, key= cluster label, val = list of cluster data points(coords)
        
        # list out data points according to their labels
        for p in DPs:
            if not p.label in clusters.keys(): # initialize cluster
                clusters[p.label] = [p.coord]
            else:
                clusters[p.label].append(p.coord)
        # update centers
        for i,c in enumerate(centers):
            if i in clusters.keys(): # check if cluster has members
                coords = np.array(clusters[i])
                c.coord = np.mean(coords,axis=0)
        self.centers = centers
        
    def labels_(self):
        '''
        returns a np.array(n_samples,) of cluster labels
        '''
        # initialize
        DPs = self.DPs
        
        labels = []
        for p in DPs:
            labels.append(p.label)
        labels = np.array(labels)
        
        return(labels)
        
    def centers_(self):
        '''
        returns the center coordinates in np.array(n_centers * n_dimensions) format
        '''
        # initialize
        centers = self.centers

        coord_list = []
        for c in centers:
            coord_list.append(c.coord)
        return(np.array(coord_list))
        
        
        
        
        
    
## Functions


def setSeeds(DPs,centers):
    '''
    Set up initial centers
    '''
    # initialize
    # DPs = self.DPs
    # k = self.k
    # initSeeds = self.initSeeds
    
    centers = []
    for i in range(k):
        c = DataPoint(index=i,isCenter=True)
        c.label = int(i)
        centers.append(c)
    
    if initSeeds == None: # generate starting points
        N = len(DPs) # number of data points
        indices = np.random.randint(10,size = k)
        for i in range(k):
            centers[i].coord = DPs[indices[i]].coord
        
    else: # use initSeeds
        for i in range(k):
            centers[i].coord = initSeeds[i]
    
    return(centers)
    # self.centers = centers
    

def cluster(DPs,centers):
    '''
    Runs the algorithm
    ----------
    '''
    # initialize
    # max_iter = self.max_iter
    
    # initialize centers
    setSeeds(DPs,centers)
    
    # start iterations:
    for i in range(max_iter):
        assign_labels(DPs,centers)
        update_centers(DPs,centers)
    
    return(DPs,centers)    

def assign_labels(DPs,centers):
    '''
    Assignment step
    '''
    # # initialize
    # DPs = self.DPs
    # centers = self.centers
    # dist = self.dist
    from scipy.spatial import distance
    dist = distance.minkowski
    
    for p in DPs:
        dists = [] # list of distances to centers
        for c in centers:
            d = dist(p.coord,c.coord)
            dists.append(d)
        dists = np.array(dists)
        index = np.argmin(dists) # find the closest center
        p.label = centers[index].label # assign cluster label
    
    return(DPs)
    # self.DPs = DPs # update


def update_centers(DPs,centers):
    '''
    Update step
    '''
    # # initialize
    # DPs = self.DPs
    # centers = self.centers
    # dist = self.dist
    clusters = {} # dictionary of lists, key= cluster label, val = list of cluster data points(coords)
    
    # list out data points according to their labels
    for p in DPs:
        if not p.label in clusters.keys(): # initialize cluster
            clusters[p.label] = [p.coord]
        else:
            clusters[p.label].append(p.coord)
    # update centers
    for i,c in enumerate(centers):
        if i in clusters.keys(): # check if cluster has members
            coords = np.array(clusters[i])
            c.coord = np.mean(coords,axis=0)
    # self.centers = centers    
    return(centers)

    
def labels_(DPs):
    '''
    returns a np.array(n_samples,) of cluster labels
    '''
    # initialize
    # DPs = self.DPs
    
    labels = []
    for p in DPs:
        labels.append(p.label)
    labels = np.array(labels)
    
    return(labels)    
    
