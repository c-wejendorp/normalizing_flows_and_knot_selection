import torch
from torch.utils.data import DataLoader
from nfModel_class import *
from coupling_layer import *
from split_and_merge import *
from data_functions import *
import numpy as np
import argparse
import os

def plot_samples(model_name, num_samples):
    #load the model   
    model_path = os.path.join("trained_models",model_name)
    checkpoint = torch.load(model_path)   

    checkpoint = torch.load(model_path)    
    model = nfModel(checkpoint['flows'], base_distribution_type = checkpoint["base_distribution_type"])
    model.load_state_dict(checkpoint['model'])    
    model.eval()
    # generate samples
    samples = model.sampleFromBaseDistribution([num_samples]+[1,28,28])
    samples = model.generativeDirection(samples)    
    # plot the samples
    # both clamped and not clamped
    show_batch_images(samples,num_samples, clamp=False,save=True)
    show_batch_images(samples,num_samples, clamp=True,save=True) 