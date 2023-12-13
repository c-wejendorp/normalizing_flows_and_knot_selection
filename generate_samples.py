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
    # clamp and floor
    images = torch.clamp(samples, 0, 255.9)
    images = torch.floor(images)
    # save individual images
    # create a folder for the images
    # Create a folder for the images
    name = "_".join(args.model_name.split("_")[:2])
    save_folder = f"sampled_images/{name}"
    os.makedirs(save_folder, exist_ok=True)
    # save the images
    if args.saveformat == "pt":
        torch.save(images, f"{save_folder}/samples_{name}.pt")
    elif args.saveformat == "png":
        for i in range(args.num_samples):
            image = images[i]
            image = image.detach().numpy()
            image = np.squeeze(image)
            # save the image
            plt.imsave(f"{save_folder}/sample_{i}.png", image, cmap="gray") 
      

if __name__ == "__main__":    
    # make an argument parser
    parser = argparse.ArgumentParser(description='Load a trained model generate samples from it')    
    #parser.add_argument('--model_name', type=str, default='', help='name of the pt made by create_flows.py')
    parser.add_argument('--model_name', type=str, default='flow_1_logistic_seed_30.pt_all_data_epoch_89.pth', help='name of the model to load')
    parser.add_argument('--model_folder', type=str, default='trained_models', help='path to the model')
    parser.add_argument('--num_samples', type=int, default=10000, help='number of samples to generate')
    parser.add_argument('--saveformat', type=str, default='pt', help='type of file to save the images as, pt or png')    
    
    # parse the arguments
    args = parser.parse_args()
    generate_samples(args)