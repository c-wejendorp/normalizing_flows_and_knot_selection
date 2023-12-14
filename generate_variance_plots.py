import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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
    # scale to be between 0 and 1
    images_org = (images_org.float()/255).squeeze()
    samples_flow_1 = (samples_flow_1.float()/255).squeeze()
    samples_flow_2 = (samples_flow_2.float()/255).squeeze()
    #squeeze the images


    variance_images = torch.var(images_org.float(), dim=0)
    variance_flow_1 = torch.var(samples_flow_1, dim=0)
    variance_flow_2 = torch.var(samples_flow_2, dim=0)
    # Visualize the variance in subplots
    fig, axs = plt.subplots(1, 3)

    # set the min and max values for the colorbar
    vmin = min(variance_images.min(), variance_flow_1.min(), variance_flow_2.min())
    vmax = max(variance_images.max(), variance_flow_1.max(), variance_flow_2.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    shrink_value = 0.4
    # Original
    im0 = axs[0].imshow(variance_images.detach().numpy(), norm=norm)
    axs[0].set_title("Original")
    fig.colorbar(im0, ax=axs[0], orientation='vertical', shrink=shrink_value)
    # Flow 1
    im1 = axs[1].imshow(variance_flow_1.detach().numpy(), norm=norm)
    axs[1].set_title("Flow 1")
    fig.colorbar(im1, ax=axs[1], orientation='vertical', shrink=shrink_value)
    # Flow 2
    im2 = axs[2].imshow(variance_flow_2.detach().numpy(), norm=norm)
    axs[2].set_title("Flow 2")
    fig.colorbar(im2, ax=axs[2], orientation='vertical', shrink=shrink_value)

    #remove ticks form all subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    # reduce whitespace
    fig.tight_layout()
    # add a title close to the figure
    fig.suptitle("Pixel Variance for Pixels Scaled to [0,1]", y=0.8)
    
    # save the figure in the figures folder in a subfolder called variance_plots
    # make the folder if it does not exist
    os.makedirs("figures/variance_plots", exist_ok=True)
    plt.savefig("figures/variance_plots/variance_plots.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_variance()

   
    




