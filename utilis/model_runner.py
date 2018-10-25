from utilis.preprocess import load_data, preprocess
from utilis.cluster import Clustering
from utilis.evaluation import Evaluate
from utilis.visualization import  plot_clustering

class Model_runner(object):
    def __init__(self):
       
        self.config = {}
        self.config['data_name'] = '3_14_T110c'
        self.config['load_data'] = {
            'file_t1':'data/T1_point.txt',
             'file_t2':'data/T2_point.txt',
             'file_fvol':'data/T1T2_3_14_T110c.txt' # T110c/ as_received
        }
        self.config['preprocess'] = {
            'min_t1t2_ratio':0.5, 
            't1_max': 600, 
            'fc':0.1, 
            'variation_length': 0.01,
            'save':False,
            'reload': True,
            'file_xtrain':'data_process/X_train_1_223_T110c.txt',
            'file_xmanifold':'data_process/X_manifold_1_223_T110c.txt'
        }
        self.config['clustering']={
            'model_names' : ['GMM','kmean'],
            'n_clusters': 5,
            'adjust_label': True,
            'save_model':False,
            'file_dir':'model/'
        } 
        
        self.config['evaluate']={
        'distance_min':0.05,
        'point_min': 10 #0.05, point_min = 50   
        }
        
        self.data_name = self.config['data_name']
        self.config['clustering']['file_dir'] += self.data_name
        self.t1_domain = None
        self.t2_domain = None
        self.t2_grid = None
        self.t1_grid = None
        self.f_grid = None
        self.X_train = None 
        self.X_manifold = None
        
        
    def load_data(self):
        self.t1_grid, self.t2_grid, self.f_grid = load_data(self.config['load_data'])
        return self
    
    def preprocess(self):
        self.X_train, self.X_manifold = preprocess(self.t2_grid, self.t1_grid, self.f_grid, 
            self.config['preprocess'])
        return self
    
    def fit(self):
        clustering = Clustering(self.X_train, self.config['clustering'])
        clustering.model_init()
        clustering.fit()
        self.model_names = clustering.model_names
        self.y_preds = clustering.y_preds
        return self
    
    def evaluate(self):
        evalu = Evaluate(self.model_names, self.X_train, self.y_preds, self.config['evaluate'])
        evalu.fit()
        self.metrics = evalu.metrics
        self.boundary_points = evalu.boundary_points