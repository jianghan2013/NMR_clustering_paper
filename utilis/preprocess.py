import numpy as np


def load_data(config, verbose=0):
    # load raw t1,t2 and fluid volume data from file
    file_t1 = config['file_t1']
    file_t2 = config['file_t2']
    file_fvol =  config['file_fvol']
    t1_domain = np.loadtxt(file_t1) # 1d minimum points
    t2_domain = np.loadtxt(file_t2) # 1d minimum points
    f_grid = np.loadtxt(file_fvol) # 2d grid points
    t2_grid,t1_grid = np.meshgrid(t2_domain, t1_domain) # 2D grid points
    # check size
    assert(f_grid.size == t2_grid.size)
    assert(f_grid.size == t1_grid.size) 
    increment_t1 = np.log10(t1_domain[0]) - np.log10(t1_domain[1])
    increment_t2 = np.log10(t2_domain[1]) - np.log10(t2_domain[2])
    if verbose:
        print('file info, Grid shape of [t1,t2]: {}, log(t1) incremental: {}, log(t2) incremental: {}'.format(
            f_grid.shape, round(increment_t1,5), round(increment_t2,5)))
    return t1_grid, t2_grid, f_grid
    
def preprocess(t2_grid, t1_grid, f_grid, config, verbose=0):
    if config['reload']:
        file_xtrain = config['file_xtrain']
        file_xmanifold = config['file_xmanifold']
        if verbose:
            print('load X_trian, X_manifold')
        X_train, X_manifold = np.loadtxt(file_xtrain), np.loadtxt(file_xmanifold)
    else:
        X_train, X_manifold = to_density(t2_grid, t1_grid, f_grid, config, verbose)
    return X_train, X_manifold

def to_density(t2_grid, t1_grid, f_grid, config, verbose=0):
    """
    Perform 4-steps preprocessing:
        1. Crop the data by selecting t1_grid > t2_grid * min_t1t2_ratio and t1_grid  < max of t1 
        2. Select f_grid > minimum of f = T1T2_max * ratio, and normalize it. 
        3. Log transform of t1_grid and t2_grid 
        4. Convert the 3-D (t1,t2,f) data to 2-D (t1,t2) data, where the frequency is 
            represented by number count of (t1,t2) grid point.
    
    Args:
        t2_grid: np.array, 2-D t2 grid points
        t1_grid: np.array, 2-D t1 grid points
        f_grid: np.array, 2-D fluid volume grid points
    
    Returns:
        X_train: np.array, 2-D (t2,t1) for clustering training
        X_manifold: np.array, 3-D (t2, t1, f_norm), 3-D data before 
    """
    # set config
    min_t1t2_ratio=config['min_t1t2_ratio'] 
    t1_max = config['t1_max'] 
    fc=config['fc'] 
    variation_length = config['variation_length'] 
    save = config['save']
    file_xtrain = config['file_xtrain']
    file_xmanifold = config['file_xmanifold']
    
    # 1. cropping 
    f_max = np.max(f_grid) 
    f_min = fc * f_max
    id_grid_select =  (t1_grid >= t2_grid * min_t1t2_ratio ) & ( t1_grid < t1_max) & (f_grid > f_min) 
    
    # 2. Normalization
    f_norm = np.round(np.array(f_grid)/(f_min)) # convert to number
    f_norm[~id_grid_select]=0
    if verbose:
        sum_raw = np.sum(f_grid) 
        sum_select = np.sum(f_grid[id_grid_select])
        print('Normalization : raw fluid volume {} , select fluid volume {}, their ratio {}'.format(
            round(sum_raw,5), round(sum_select,5), round(sum_select/sum_raw,3)))

    # flatten to 1D 
    t2_flat = np.reshape(t2_grid, (-1,1)).flatten()
    t1_flat =np.reshape(t1_grid, (-1,1)).flatten()
    f_flat = np.reshape(f_grid, (-1,1)).flatten()
    
    # manifolold: (t2,t1,f)
    f_norm_flat = np.reshape(f_norm, (-1,1)).flatten().astype(int) # conver to integer
    id_nonzero = f_norm_flat > 0
    
    # 3. log tranformation and get manifold
    X_manifold = np.vstack([np.log10(t2_flat[id_nonzero]), np.log10(t1_flat[id_nonzero]), f_flat[id_nonzero]]).T
    
    #X_manifold = pd.DataFrame(X_manifold, columns=['t2','t1','fnorm'])
    # 3-D data

    # 4. convert 3-D manifold to 2D density (t2,t1) for training clustering
    t2_den = []
    t1_den = []
    for i, f_norm in enumerate(f_norm_flat):
        if f_norm >0:
            t2_i = np.array([np.log10(t2_flat[i])] * f_norm) + np.random.randn(1, f_norm) * variation_length
            t2_i.flatten()
            t1_i = [np.log10(t1_flat[i])] * f_norm + np.random.randn(1, f_norm) * variation_length
            t1_i.flatten()
            t2_den.extend(t2_i[0])
            t1_den.extend(t1_i[0])
        #break
    X_train = np.vstack([np.array(t2_den), np.array(t1_den)]).T # 2-D data
    if verbose:
        print('X_trian shape:{}, X_manifold shape: {}'.format(X_train.shape, X_manifold.shape))
    
    if save:
        if verbose:
            print('X_trian, X_manifold saved')
        np.savetxt(file_xtrain,X_train)
        np.savetxt(file_xmanifold, X_manifold)
    return X_train, X_manifold