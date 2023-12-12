import torch
from nfModel_class import *
from coupling_layer import *
from split_and_merge import *
from data_functions import *
import numpy as np
import os

def createFlows(base_distributions=["normal","logistic"],list_of_splits=None,img_shape=(1,28,28),seeds=[42]): 
    # create folder for models
    if not os.path.exists("untrained_models"):
        os.makedirs("untrained_models")

    for name in ["flow_1","flow_2"]:    
        for base_type in base_distributions:
            for seed in seeds:
                # set seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                # create model
            
                sub_flows = []
                # add preprocessing flow
                sub_flows.append(scaleAll(img_shape,preprocess=True))

                if name == "flow_1":
                    for type in list_of_splits[0]:
                        sub_flows.append(Split(split_type=type,input_shape=img_shape))                    
                        sub_flows.append(couplingLayer(img_shape,num_hidden_layers=5,num_hidden_units=1000,scale=False,init_zeros=True))
                        sub_flows.append(Merge(split_type=type,input_shape=img_shape))

                    sub_flows.append(scaleAll(img_shape))
                    
                elif name == "flow_2":
                    for type in list_of_splits[1]:
                        sub_flows.append(Split(split_type=type,input_shape=img_shape))                    
                        sub_flows.append(couplingLayer(img_shape,num_hidden_layers=5,num_hidden_units=1000,scale=False,init_zeros=True))
                        sub_flows.append(Merge(split_type=type,input_shape=img_shape))

                    sub_flows.append(scaleAll(img_shape))                         

                model = nfModel(sub_flows, base_distribution_type = base_type)
                # save model
                dict_to_save = {
                    "model": model.state_dict(),
                    "flows": sub_flows,
                    "base_distribution_type": base_type,
                    "img_shape": img_shape,        }

                torch.save(dict_to_save, f"untrained_models/{name}_{base_type}_seed_{seed}.pt")

if __name__ == "__main__":
    base_distributions = ["normal","logistic"]
    split_for_flow_1 = ["columns","columns_odd","top_bottom","bottom_top","checkerboard","checkerboard_odd"]
    split_for_flow_2 = ["random"] * 6
    list_of_splits = [split_for_flow_1,split_for_flow_2]
    seeds = [15,30,45]    
    createFlows(base_distributions=base_distributions,list_of_splits=list_of_splits,seeds=seeds)