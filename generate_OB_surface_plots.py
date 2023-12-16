import torch
from torch.utils.data import DataLoader
from nfModel_class import *
from coupling_layer import *
from split_and_merge import *
from data_functions import *
import numpy as np
import torchvision
from torchvision.transforms.functional import InterpolationMode
import os
from select_and_evaluate_knots import uniform_selection, non_uniform_selection
from spline_functions import *

def plot_surfaces(num_samples,num_knots=10,seed=32,scale=None):
    if num_knots % 2 == 0 and num_knots >= 4 and num_knots <= 22:
        surface_index = (num_knots - 4) // 2
    else:
        assert False, "Number of knots must be even and at least 4 and at most 22" 
    
    #load test set
    test_set = createDataSet(1)[1]
    test_set = test_set.data.squeeze().float()/255

    # load surface for the three different models and uniform
    # select number of knots. 2 is index 0, 4 is index 1, etc.
    if scale is None:      
        surfaces = {"Images": test_set,
                    "Uniform": torch.load("OB_surfaces/test_OB_surfaces_uniform_Original.pt")[surface_index],    #uniform is the same for all models
                    "org trainset": torch.load("OB_surfaces/test_OB_surfaces_non_uniform_Original.pt")[surface_index],
                    "flow_1": torch.load("OB_surfaces/test_OB_surfaces_non_uniform_Flow_1.pt")[surface_index],
                    "flow_2": torch.load("OB_surfaces/test_OB_surfaces_non_uniform_Flow_2.pt")[surface_index]}
    else:
        # upscale the resulution of the test set
        test_set = torchvision.transforms.Resize((test_set.shape[1]*scale,test_set.shape[2]*scale),interpolation=InterpolationMode.NEAREST)(test_set)
        surfaces = {"Images": test_set,
                    "Uniform": torch.load(f"OB_surfaces/test_OB_surfaces_uniform_Original_scale_{scale}.pt")[surface_index],
                    "org trainset": torch.load(f"OB_surfaces/test_OB_surfaces_non_uniform_Original_scale_{scale}.pt")[surface_index],
                    "flow_1": torch.load(f"OB_surfaces/test_OB_surfaces_non_uniform_Flow_1_scale_{scale}.pt")[surface_index],
                    "flow_2": torch.load(f"OB_surfaces/test_OB_surfaces_non_uniform_Flow_2_scale_{scale}.pt")[surface_index]}
  
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
    # afjust title depending on we scal 
    if scale is None:
        fig.suptitle(f"Surface plots for different models with {num_knots} knots, 28x28", fontsize=16)
    else:
        fig.suptitle(f"Surface plots for different models with {num_knots} knots, 112x112", fontsize=16)
    # save figure
    os.makedirs("figures/surface_plots", exist_ok=True)
    if scale is None:        
        plt.savefig(f"figures/surface_plots/OB_surfaces_{num_knots}_knots.png", dpi=300, bbox_inches='tight')
    else:       
        plt.savefig(f"figures/surface_plots/OB_surfaces_{num_knots}_knots_scale_{scale}.png", dpi=300, bbox_inches='tight')


def plot3D(num_knots=10,seed=32,knot_type="uniform",degree=3,grid=1000):
    #load test set
    train_set,test_set = createDataSet(1)
    train_set = train_set.data.squeeze().float()/255
    test_set = test_set.data.squeeze().float()/255
    img_shape = train_set.shape[1:]
    # select knots
    if knot_type == "uniform":
        x_knots, y_knots = uniform_selection(num_knots, img_shape, degree)
    else:
        x_knots, y_knots = non_uniform_selection(train_set,num_knots, img_shape, degree)

    L1 = len(x_knots) - (2*degree) -1 
    L2 = len(y_knots) - (2*degree) -1
        
    # create the B-spline basis matrices
    A = B_spline_matrix(torch.arange(img_shape[0]), degree + 1 + L1 -1, degree, x_knots)
    B = B_spline_matrix(torch.arange(img_shape[1]), degree + 1 + L2 -1, degree, y_knots)
    A = gram_schmidt(A,use_QR=False)
    B = gram_schmidt(B,use_QR=False)

    
    A_scaled = B_spline_matrix(torch.linspace(0,img_shape[0],img_shape[0]*grid), degree + 1 + L1 -1, degree, x_knots)
    B_scaled = B_spline_matrix(torch.linspace(0,img_shape[1],img_shape[1]*grid), degree + 1 + L2 -1, degree, y_knots)
    A_scaled = gram_schmidt(A_scaled,use_QR=False)
    B_scaled = gram_schmidt(B_scaled,use_QR=False)      

    #select a random image from the test set    
    np.random.seed(seed)
    image_indices = np.random.choice(len(test_set), 1)
    image = test_set[image_indices]

            
    AG = torch.matmul(torch.transpose(A, 0, 1), image)
    AGB = torch.matmul(AG, B)
    C = AGB        
    # now find the surfaces of the test set
    z_splines_test = torch.matmul(A_scaled, C) @ torch.transpose(B_scaled, 0, 1)
    # select a random image from the test set  

    #show the image
    plt.imshow(image.squeeze().detach().numpy(),cmap='gray')
    plt.show()  
    
    # plot the image in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
        # Create a meshgrid for 3D plot
    x_values = np.linspace(0, img_shape[0], img_shape[0]*grid)
    y_values = np.linspace(0, img_shape[1], img_shape[1]*grid)

    X, Y = np.meshgrid(x_values, y_values)
    # Plot the surface
    ax.plot_surface(X, Y, z_splines_test.squeeze().detach().numpy(), cmap='viridis')
    plt.show()
 

if __name__ == "__main__":
    # 3d plot
    #plot3D(num_knots=22,seed=32,knot_type="uniform",degree=3,grid=100)
    # 2d plots
    for num_knots in range(4,24,2):    
        plot_surfaces(10,num_knots=num_knots,seed=17)
        plot_surfaces(10,num_knots=num_knots,seed=17,scale=4)
    
   
