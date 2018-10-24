from sklearn import cluster
from sklearn import mixture
from sklearn.externals import joblib
import numpy as np
import pandas as pd

class Clustering(object):
    def __init__(self, X_train, config, verbose=0):
        self.model_names = config['model_names']
        self.n_clusters = config['n_clusters']
        self.adjust_label = config['adjust_label']
        self.save_model = config['save_model'] 
        self.file_dir = config['file_dir']    
        self.verbose = verbose
        
        self.X_train = X_train
        self.models = None
        self.y_preds = {}
    def model_init(self):
        self.models = model_init(model_names = self.model_names, n_clusters=self.n_clusters, verbose=self.verbose)
        return self
    def fit(self):
        for model_name in self.models:
            if self.verbose:
                print('model_name: {}'.format(model_name))
            y_pred =  clustering_(model = self.models[model_name], 
                X_train = self.X_train, adjust_label = self.adjust_label, verbose=self.verbose)
            self.y_preds[model_name]= y_pred
            if self.save_model:
                model_file = self.file_dir + '_{}_ncluster_{}.pkl'.format(model_name,self.n_clusters)
                joblib.dump(self.models[model_name], model_file)
        return self
        
def model_init(model_names = [], n_clusters=5, verbose=0):
    """
    To initate clustering models
    """
    # GMM
    model_names_all =  ['birch','complete','GMM','kmean','spectral','ward']
    if model_names == []:
        model_names = model_names_all
#     for model_name in model_names:
#         if model_name not in model_names_all:
#             raise ValueError('your model {} is not in the model_list {}'.format(model_name, model_names_all)
                             
    if verbose:
        print('all models {}, selected models {}'.format(model_names_all, model_names))
        print(model_names_all)
                             
                                 
    # model setting  
    birch_args = {'n_clusters': n_clusters}
    complete_args = {'n_clusters': n_clusters, 'linkage': 'complete'}
    GMM_args = {'n_components': n_clusters, 'covariance_type' : 'full', 'max_iter' : 2000,'tol':1e-4}
    kmean_args = {'n_clusters': n_clusters, 'n_init': 20}
    spectral_args = {'n_clusters':n_clusters, 'eigen_solver':'arpack', 'affinity':'nearest_neighbors'}
    ward_args = {'n_clusters': n_clusters, 'linkage': 'ward'}
    
    models_init = {e:None for e in model_names_all}
    models_init['birch'] = cluster.Birch(**birch_args)
    models_init['complete'] = cluster.AgglomerativeClustering(**complete_args)                      
    models_init['GMM'] = mixture.BayesianGaussianMixture(**GMM_args)
    models_init['kmean'] = cluster.KMeans(**kmean_args)
    models_init['spectral'] = cluster.SpectralClustering(**spectral_args)
    models_init['ward'] = cluster.AgglomerativeClustering(**ward_args)
    
    # select model
    models = {name: models_init[name] for name in model_names}
    if verbose:
        print('*** model settings ***')
        print(models.keys)
    return models

def approx_centers(X_train,y_pred):
    # sometime need to calculate the approximate center
    n_components = max(y_pred)+1
    centers = []
    for i in range(n_components):
        x,y = np.mean(X_train[y_pred==i], axis=0)
        centers.append([x,y])
    return np.array(centers)

def label_mapping(centers, labels, weights = None ):
    """ To reassign the label based on the position. 
        Wihtout this, clustering algorithm will always randomly assign the label values 
    
    """
    df_data = pd.DataFrame(centers, columns = ['T2','T1'])
    df_data['labels_old'] = labels
    if weights is not None:
        df_data['weights'] = weights
    for i in range(df_data.shape[0]):
        if df_data.loc[i,'T2'] < -0.5:
            #print(df_data.loc[i,'T2'])
            df_data.loc[i,'T1/T2'] = df_data.loc[i,'T1']
        else:
            df_data.loc[i,'T1/T2'] = df_data.loc[i,'T2']+10
    df_data.sort_values('T1/T2', inplace=True)
    df_data['labels_new'] = range(df_data.shape[0])
    mapping = {label: i for i,label in enumerate(df_data['labels_old'])} 
    #print(df_data)
    return mapping, df_data[['T2','T1','labels_new']].values

def clustering_(model, X_train, adjust_label = True, verbose=0):
    # single model 
    model.fit(X_train)
    if hasattr(model, 'labels_'):
        y_pred = model.labels_.astype(np.int)
    else:
        y_pred = model.predict(X_train)
        
    if verbose:
        print('clustering succuss' )    
    # remapping 
    if adjust_label:
        if verbose:
            print('adjust labels (remapping)')
        cluster_centers = approx_centers(X_train, y_pred)
        n_components = max(y_pred)+1
        labels = list(range(n_components))
        mapping, cluster_centers = label_mapping(cluster_centers, labels)
        y_pred_new = np.array([mapping[y_old] for y_old in y_pred])
        return y_pred_new
    return y_pred 

