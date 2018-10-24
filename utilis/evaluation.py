### evaluation
import numpy as np
from sklearn.linear_model import LinearRegression

class Evaluate(object):
    def __init__(self, model_names, X_train, y_preds, config,verbose=0):
        self.distance_min = config['distance_min']
        self.point_min = config['point_min'] #0.05, point_min = 50
        self.model_names = model_names
        self.X_train= X_train
        self.y_preds = y_preds
        self.verbose = verbose
        
        self.metrics = {'ratios':{}, 'slopes': {}, 'inters':{}, 'slopes_raw':{}}
        self.boundary_points = {}
        
    def fit(self):
        for model_name in self.model_names:
            ratios = get_ratio_range(self.X_train, self.y_preds[model_name])
            slopes, inters, slopes_raw, boundaries   =  get_boundary_and_slope(self.X_train, self.y_preds[model_name], self.distance_min, self.point_min)
            self.metrics['ratios'][model_name] =ratios
            self.metrics['slopes'][model_name] = slopes
            self.metrics['slopes_raw'][model_name] = slopes_raw
            self.metrics['inters'][model_name] = inters
            self.boundary_points[model_name] = boundaries
            if self.verbose:
                print('model_name {}, metrics ratios {}, slopes {}, inters{}'.format(model_name, 
                    self.metrics['ratios'][model_name], self.metrics['slopes'][model_name],
                    self.metrics['inters'][model_name]))
        return self

def get_ratio_range(X_train, y_pred):
    """
    Compute range ratio index 
    """
    range_ratios=[]
    n_components = max(y_pred)+1
    for i in range(n_components):
        X_train_i = X_train[y_pred==i]
        T2_v = 10**(X_train_i[:,0])
        T1_v = 10**(X_train_i[:,1])
        range_ratio = (np.max(T1_v/T2_v)/np.min(T1_v/T2_v))
        range_ratios.append(range_ratio)
    return range_ratios

def get_boundary_from_two_clusters_(cluster_a, cluster_b, distance_min = 0.05):
    # cluster_a: shape(n,2)
    # cluster_b: shape(n,2)
    id_a =set()
    id_b =set()# the pair of row id (i,j), i is for cluster_a and j is for cluster_b
    for i in range(cluster_a.shape[0]):
        #i = 0
        clsuter_a_i = cluster_a[i,:]
        distance_list = np.sqrt( (clsuter_a_i[0]-cluster_b[:,0])**2 + (clsuter_a_i[1]-cluster_b[:,1])**2)
        distance_ = np.amin(distance_list) # mini distance
        if distance_ < distance_min:
            j = np.argmin(distance_list)
            id_a.add(i)
            id_b.add(j)
    if len(id_a) == 0 and len(id_a) == 0:
        return []
    else:
        id_a = list(id_a)
        id_b = list(id_b)
        id_a.sort()
        id_b.sort()
        boundary_points = np.vstack(  (cluster_a[id_a,:],cluster_b[id_b,:] )   )
    return boundary_points

def get_boundary_and_slope(X_train, y_pred, distance_min=0.05, point_min = 50):
    # point_min minimum point for the boundary points
    # get the decision boundary and their slopes
    boundary_list = [] # contains all boundary points
    slope_raw_list = []
    angle_diff_list = [] # contains the slope for that boundary
    inter_list = []
    n_components = max(y_pred)+1
    data_all = [X_train[y_pred==i] for i in range(n_components)] # get each cluster points
    for i in range(n_components-1):
        for j in range(i+1, n_components):
            cluster_a = data_all[i]
            cluster_b = data_all[j]
            boundary_points = get_boundary_from_two_clusters_(cluster_a, cluster_b,distance_min = distance_min)
            if len(boundary_points) > point_min:
                boundary_list.append(boundary_points)
                # linear regression
                lr_ = LinearRegression()
                X_ = boundary_points[:,0].reshape(-1,1)
                y_ = boundary_points[:,1]
                lr_.fit(X_,y_)
                slope = lr_.coef_[0]/np.pi*180
                inter = lr_.intercept_
                slope_raw_list.append(slope)
                inter_list.append(inter)
                diff_slope = min(abs(slope-45),(180+slope-45)) # when angle < -45, it should be the complementary angle
                 
                angle_diff_list.append(diff_slope) # normalize slope
                
    return angle_diff_list, inter_list, slope_raw_list, boundary_list  

