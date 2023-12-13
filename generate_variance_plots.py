import numpy as np
import torch
import matplotlib.pyplot as plt
from data_functions import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


def plot_variance():
    #load the orginal dataset
    train_set, _ = createDataSet(1)
    images_org = train_set.data
    samples_flow_1 = torch.load("sampled_images/flow_1/samples_flow_1.pt")
    samples_flow_2 = torch.load("sampled_images/flow_2/samples_flow_2.pt")
    # Find the variance
    variance_images = torch.var(images_org.float(), dim=0)
    variance_flow_1 = torch.var(samples_flow_1, dim=0)
    variance_flow_2 = torch.var(samples_flow_2, dim=0)
    # Visualize the variance in subplots
    fig, axs = plt.subplots(1, 3)

    # Original
    im0 = axs[0].imshow(variance_images.detach().numpy())
    axs[0].set_title("Original")
    # Flow 1
    im1 = axs[1].imshow(variance_flow_1[0].detach().numpy())
    axs[1].set_title("Flow 1")
    # Flow 2
    im2 = axs[2].imshow(variance_flow_2[0].detach().numpy())
    axs[2].set_title("Flow 2")        
    # Add a single colorbar for all subplots
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im2, cax=cax)
    # reduce whitespace
    fig.tight_layout()
    # add a title close to the figure
    fig.suptitle("Pixel Variance", y=0.8)
    
    # save the figure in the figures folder in a subfolder called variance_plots
    # make the folder if it does not exist
    os.makedirs("figures/variance_plots", exist_ok=True)
    plt.savefig("figures/variance_plots/variance_plots.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_variance()

   
    




