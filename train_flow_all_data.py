import torch
from torch.utils.data import DataLoader
from nfModel_class import *
from coupling_layer import *
from split_and_merge import *
from data_functions import *
import numpy as np
import argparse
import os
from sklearn.model_selection import KFold

def train_model(args):   

    # create folder for trained models
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    
    #load the model   
    model_path = os.path.join(args.model_folder,args.model_name)
    inital_checkpoint = torch.load(model_path)
    model = nfModel(inital_checkpoint["flows"], base_distribution_type = inital_checkpoint["base_distribution_type"],)
    

    # choose optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)        

    # load the data
    train_set, test_set = createDataSet(args.digit)

    # train on all data
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # test on all data
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)   
    
    print(f'{args.model_name}_all_data')   
   
    train_loss_hist = []
    test_loss_hist = []

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        # train the model             
        model.train()
        curent_train_loss = []
        for batch, _ in train_loader:                              

            loss  = model.forwardKL(batch)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curent_train_loss.append(loss.detach().to('cpu').numpy())
        # keep track of the training loss over epochs    
        train_loss_hist.append(np.mean(curent_train_loss))           


        # evaluate the model
        model.eval()
        with torch.no_grad():
            # loss on validation set
            val_loss = []
            for batch, _ in test_loader:                                      
                loss  = model.forwardKL(batch)  
                val_loss.append(loss.detach().to('cpu').numpy())
            # keep track of the validation loss over epochs
            test_loss_hist.append(np.mean(val_loss))

        #pring the current train and validation loss
        print(f"Train loss: {train_loss_hist[-1]}")
        print(f"Test loss: {test_loss_hist[-1]}")

        # save the model and flow structure every 50 epochs
        if (epoch+1) % 50 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'flows': model.flows,
                'loss_hist': train_loss_hist,
                'test_loss_hist': test_loss_hist,
                'base_distribution_type': inital_checkpoint["base_distribution_type"],
            }
            torch.save(checkpoint, f'trained_models/{args.model_name}_all_data_epoch_{epoch}.pth')

        # save the model and flow structure after all epochs

    checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'flows': model.flows,
                'loss_hist': train_loss_hist,
                'test_loss_hist': test_loss_hist,
                'base_distribution_type': inital_checkpoint["base_distribution_type"],
            }
    torch.save(checkpoint, f'trained_models/{args.model_name}_fold_all_data_epoch_{epoch}.pth')

if __name__ == "__main__":
    
    # make an argument parser
    parser = argparse.ArgumentParser(description='Train a normalizing flow model.')
    parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')       
    #parser.add_argument('--model_name', type=str, default='', help='name of the pt made by create_flows.py')
    parser.add_argument('--model_name', type=str, default='flow_1_logistic_seed_15.pt', help='name of the pt made by create_flows.py')
    parser.add_argument('--model_folder', type=str, default='untrained_models', help='path to the model')
    parser.add_argument('--digit', type=int, default=1, help='digit to train on')
    
    # parse the arguments
    args = parser.parse_args()
    train_model(args)



