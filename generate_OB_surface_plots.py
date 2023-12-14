import torch
from torch.utils.data import DataLoader
from nfModel_class import *
from coupling_layer import *
from split_and_merge import *
from data_functions import *
import numpy as np
import os

def plot_surfaces(num_samples,num_knots=10,seed=32):
    if num_knots % 2 == 0 and num_knots >= 4 and num_knots <= 22:
        surface_index = (num_knots - 4) // 2
    else:
        assert False, "Number of knots must be even and at least 4 and at most 22" 
    
    #load test set
    test_set = createDataSet(1)[1]
    test_set = test_set.data.squeeze().float()/255

    # load surface for the three different models and uniform
    # select number of knots. 2 is index 0, 4 is index 1, etc.      
    surfaces = {"Images": test_set,
                "Uniform": torch.load("OB_surfaces/test_OB_surfaces_uniform_Original.pt")[surface_index],    #uniform is the same for all models
                "org trainset": torch.load("OB_surfaces/test_OB_surfaces_non_uniform_Original.pt")[surface_index],
                "flow_1": torch.load("OB_surfaces/test_OB_surfaces_non_uniform_Flow_1.pt")[surface_index],
                "flow_2": torch.load("OB_surfaces/test_OB_surfaces_non_uniform_Flow_2.pt")[surface_index]}

    #select random images from test set
    # set seed 
    np.random.seed(seed)
    image_indices = np.random.choice(len(test_set), num_samples)
    # plot the surfaces for the different models next to each other in a subplot
    fig, axs = plt.subplots(5, num_samples, figsize=(20, 10))
    for key,value in surfaces.items():
        surface = torch.index_select(value, 0, torch.tensor(image_indices))
        for plot_idx,i in enumerate(image_indices):
            axs[list(surfaces.keys()).index(key), plot_idx].imshow(surface[plot_idx].detach().numpy(),cmap='gray')
    # add titles
    #for ax, col in zip(axs[0], image_indices):
    #    ax.set_title(col)
    # add labels
    for ax, row in zip(axs[:,0], surfaces.keys()):
        ax.set_ylabel(row, rotation=90, size='large')
    # remove ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    # add title for whole plot
    fig.suptitle(f"Surface plots for different models with {num_knots} knots", fontsize=16)
    # save figure
    plt.savefig(f"figures/surface_plots/OB_surfaces_{num_knots}_knots.png", bbox_inches='tight', dpi=300)
    
    #plt.show()

  
    


    

if __name__ == "__main__":
    for num_knots in range(4,24,2):    
        plot_surfaces(10,num_knots=num_knots,seed=17)
    
   
