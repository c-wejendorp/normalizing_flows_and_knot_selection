import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from data_functions import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def load_samples(flow_folder):
    return (torch.load(f"sampled_images/{flow_folder}/samples_{flow_folder}.pt").float() / 255).squeeze()

def plot_flow(ax, title, variance,fig,knots=None,normalizer=None):
    im = ax.imshow(variance.detach().numpy(), norm=normalizer)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.4)

    if knots is not None:        
            [ax.axvline(x=k, color="red", linewidth=0.5) for k in knots[0]]
            [ax.axhline(y=k, color="red", linewidth=0.5) for k in knots[1]]
            
            ax.set_xticks(knots[0])
            ax.set_yticks(knots[1])
            ax.set_xticklabels(knots[0],fontsize=4)
            ax.set_yticklabels(knots[1],fontsize=4)
 
def plot_variance(knots=None):
    train_set, _ = createDataSet(1)
    images_org = (train_set.data.float() / 255).squeeze()
    
    flows = ["flow_1", "flow_2"]
    variances = [torch.var(images_org, dim=0)]+[torch.var(load_samples(flow), dim=0) for flow in flows]
    # find the maximum variance across all flows
    max_variance = max([torch.max(variance) for variance in variances]+[torch.max(torch.var(images_org, dim=0))])
    min_variance = min([torch.min(variance) for variance in variances]+[torch.min(torch.var(images_org, dim=0))])
    # create normalizer
    normalizer = Normalize(vmin=min_variance, vmax=max_variance)

    fig, axs = plt.subplots(1, len(flows) + 1)

    for i, (ax, title, variance) in enumerate(zip(axs, ["Original"] + flows, variances)):
        plot_flow(ax, title, variance, fig, knots=knots, normalizer=normalizer)
    if knots is None:
        [ax.set_xticks([]) for ax in axs]
        [ax.set_yticks([]) for ax in axs]
    
    fig.tight_layout()
    fig.suptitle("Pixel Variance for Pixels Scaled to [0,1]", y=0.8)

    os.makedirs("figures/variance_plots", exist_ok=True)
    plt.savefig("figures/variance_plots/variance_plots.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    plot_variance()
    #plot_variance(knots=[[4, 8, 12, 16, 20], [4, 8, 12, 16, 20]])


   
    




