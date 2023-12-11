import torch
from nfModel_class import *
from coupling_layer import *
from split_and_merge import *
from data_functions import *
import numpy as np
import os

def createFlows(base_distributions=["normal","logistic"],splits=None,img_shape=(1,28,28),seeds=[42]): 
    # create folder for models
    if not os.path.exists("untrained_models"):
        os.makedirs("untrained_models")

    for name in ["flow_1","flow_2"]:    
        for base_type in base_distributions:
            for seed in seeds:
                sub_flows = []
                # add preprocessing flow
                sub_flows.append(scaleAll(img_shape,preprocess=True))

                if name == "flow_1":
                    for type in splits:
                        sub_flows.append(Split(split_type=type))                    
                        sub_flows.append(couplingLayer(img_shape,num_hidden_layers=5,num_hidden_units=1000,scale=False,init_zeros=True))
                        sub_flows.append(Merge(split_type=type))

                    sub_flows.append(scaleAll(img_shape))
                    
                elif name == "flow_2":
                    pass                      

                model = nfModel(sub_flows, base_distribution_type = base_type)
                # save model
                torch.save(model.state_dict(), f"untrained_models/{name}_{base_type}_seed_{seed}.pt")

if __name__ == "__main__":
    base_distributions = ["normal","logistic"]
    split_for_flow_1 = ["columns","columns_odd","checkerboard","checkerboard_odd"]
    seeds = [32,42,52]
    createFlows(base_distributions=base_distributions,splits=split_for_flow_1,seeds=seeds)