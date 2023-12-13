import torch
from torch.utils.data import DataLoader
from nfModel_class import *
from coupling_layer import *
from split_and_merge import *
from data_functions import *
import numpy as np
import argparse
import os

#script that calculates the cross validation results


# load models and find average test and validation loss on each of the two folds
def loss_plots_cross_validation():
    for flow_name in ["flow_1","flow_2"]:
        #load the model
        for modeltype in ["logistic","normal"]:
            for fold in [0,1]:
                train_loss = []
                val_loss = []
                for seed in [15,30,45]:
                    model_name = f"{flow_name}_{modeltype}_seed_{seed}.pt_fold_{fold}_epoch_149.pth"
                    model_path = os.path.join("trained_models",model_name)
                    checkpoint = torch.load(model_path)
                    train_loss.append(checkpoint["loss_hist"])
                    val_loss.append(checkpoint["val_loss_hist"])
                # calculate the average loss and std and plot it
                train_loss = np.array(train_loss)
                val_loss = np.array(val_loss)
                train_loss_mean = np.mean(train_loss,axis=0)
                val_loss_mean = np.mean(val_loss,axis=0)
                train_loss_std = np.std(train_loss,axis=0)
                val_loss_std = np.std(val_loss,axis=0)
                # plot the results
                # keep the y axis the same for all plots
                y_max = 6000
                plt.ylim(0,y_max)

                plt.plot(train_loss_mean,label="Mean Training error")
                plt.plot(val_loss_mean,label="Mean Validation error")
                plt.fill_between(np.arange(len(train_loss_mean)),train_loss_mean-train_loss_std,train_loss_mean+train_loss_std,alpha=0.3,label="Std Train Loss",)
                plt.fill_between(np.arange(len(val_loss_mean)),val_loss_mean-val_loss_std,val_loss_mean+val_loss_std,alpha=0.3,label="Std Validation Loss",)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Average loss for {flow_name} {modeltype} fold {fold}")
                plt.legend()
                plt.savefig(f"figures/loss_plots/{flow_name}_{modeltype}_fold_{fold}.png",dpi=300)
                plt.clf()
                plt.close()

def loss_plots_all_data():
    for flow_name in ["flow_1","flow_2"]:
        #load the model
        for modeltype in ["logistic","normal"]:
            train_loss = []
            val_loss = []
            for seed in [15,30,45]:
                model_name = f"{flow_name}_{modeltype}_seed_{seed}_all_data.pt"
                model_path = os.path.join("trained_models",model_name)
                checkpoint = torch.load(model_path)
                train_loss.append(checkpoint["loss_hist"])
                val_loss.append(checkpoint["val_loss_hist"])
            # calculate the average loss and std and plot it
            train_loss = np.array(train_loss)
            val_loss = np.array(val_loss)
            train_loss_mean = np.mean(train_loss,axis=0)
            val_loss_mean = np.mean(val_loss,axis=0)
            train_loss_std = np.std(train_loss,axis=0)
            val_loss_std = np.std(val_loss,axis=0)
            # plot the results
            # keep the y axis the same for all plots
            y_max = 6000
            plt.ylim(0,y_max)

            plt.plot(train_loss_mean,label="Mean Training error")
            plt.plot(val_loss_mean,label="Mean Validation error")
            plt.fill_between(np.arange(len(train_loss_mean)),train_loss_mean-train_loss_std,train_loss_mean+train_loss_std,alpha=0.3,label="Std Train Loss",)
            plt.fill_between(np.arange(len(val_loss_mean)),val_loss_mean-val_loss_std,val_loss_mean+val_loss_std,alpha=0.3,label="Std Validation Loss",)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Average loss for {flow_name} {modeltype}")
            plt.legend()
            plt.savefig(f"figures/loss_plots/{flow_name}_{modeltype}_all_data.png",dpi=300)
            plt.clf()
            plt.close()


            




                
    
