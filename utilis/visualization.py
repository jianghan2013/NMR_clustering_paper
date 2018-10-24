### plotting
from itertools import cycle, islice
import matplotlib.pyplot as plt
import numpy as np


def plot_Xtrain(X_train):
    plt.figure()
    plt.plot(X_train[:,0], X_train[:,1], 'o', markersize=2, alpha=0.1)
    plt.xlim([-2,3])
    plt.ylim([-2,3])
    plt.xlabel('log(T2) (ms)')
    plt.ylabel('log(T1) (ms)')

def plot_clustering(X_train, y_pred, model_name = 'model',plot_save = False,
                plot_line = True, plot_center = False, plot_title=True):
    colors = np.array(list(islice(cycle(['b', 'y','r', 'g','purple','k','maroon','olive']),int(max(y_pred) + 1))))
    plt.figure(figsize=(5,5))
    plt.scatter(X_train[:, 0], X_train[:, 1], s=3, color=colors[y_pred], alpha=0.2)
    
    plt.xlim([-2,3])
    plt.ylim([-2,3])
    plt.xticks([-2,-1,0,1,2,3],fontsize=15)
    plt.yticks([-2,-1,0,1,2,3],fontsize=15)
    
    if plot_line:
        plt.plot([-2,3],[-2,3],'k')
        plt.plot([-2,3],[-1,4],'k')
        plt.plot([-2,3],[-0,5],'k')
    if plot_center:
        for (x,y,label) in centers:
            plt.text(x,y,int(label), fontsize=30)
            plt.plot(x,y,'k*')
   
        #plt.xlabel(r'$\log{T_2}$')
        #plt.ylabel(r'$\log{T_1}$')
    if plot_title:
        plt.title(model_name,fontsize=18)
    if plot_save:
        plt.savefig(model_name)

# plot for bd
def plot_boundary(boundary_list,model_name='sample',plot_title = True,plot_save = False):
    plt.figure(figsize=(5,5))
    for bd_points in boundary_list:
        plt.plot(bd_points[:,0],bd_points[:,1],'.')
    plt.plot([-2,3],[-2,3],'k')
    plt.plot([-2,3],[-1,4],'k')
    plt.plot([-2,3],[0,5],'k')
    plt.xlim(-2., 3.)
    plt.ylim(-2., 3.)
    plt.xlabel('log T2 (ms)')
    plt.ylabel('log T1 (ms)')
    if plot_title:
        plt.title(model_name,fontsize=18)
    if plot_save:
        plt.savefig(model_name)