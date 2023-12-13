import torch
from torch.utils.data import DataLoader
from nfModel_class import *
from coupling_layer import *
from split_and_merge import *
from data_functions import *
import numpy as np
import argparse
import os

def generate_samples(args):
    #load the model   
    model_path = os.path.join(args.model_folder,args.model_name)
    checkpoint = torch.load(model_path)   

    checkpoint = torch.load(model_path)    
    model = nfModel(checkpoint['flows'], base_distribution_type = checkpoint["base_distribution_type"])
    model.load_state_dict(checkpoint['model'])    
    model.eval()
    # generate samples
    samples = model.sampleFromBaseDistribution([args.num_samples]+[1,28,28])
    samples = model.generativeDirection(samples)    
    # plot the samples
    # create labels for the samples
    labels = torch.ones(args.num_samples)
    if args.clamp:
        show_batch_images(samples, labels, args.num_samples, clamp=True)
    else:
        show_batch_images(samples, labels, args.num_samples)

if __name__ == "__main__":    
    # make an argument parser
    parser = argparse.ArgumentParser(description='Load a trained model generate samples from it')    
    #parser.add_argument('--model_name', type=str, default='', help='name of the pt made by create_flows.py')
    parser.add_argument('--model_name', type=str, default='flow_1_logistic_seed_15.pt_fold_0_epoch_149.pth', help='name of the model to load')
    parser.add_argument('--model_folder', type=str, default='trained_models', help='path to the model')
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples to generate')
    parser.add_argument('--clamp', type=int, default=0, help='clamp the values to [0,256], 1 for yes, 0 for no')    
    
    
    # parse the arguments
    args = parser.parse_args()
    generate_samples(args)