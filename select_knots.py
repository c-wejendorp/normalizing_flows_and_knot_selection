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