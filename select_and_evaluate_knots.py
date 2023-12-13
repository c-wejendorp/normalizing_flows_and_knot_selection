import numpy as np
import torch
import matplotlib.pyplot as plt
from data_functions import *
import os
from spline_functions import *
import pandas as pd

def select_knots(train_set,test_set,degree,uniform=False):
    """
    Selects the knots for the B-spline basis functions using the k-means algorithm.
    
    Parameters:
    - dataset: The dataset to use for the k-means algorithm.
    - B-spline_order: The order of the B-spline basis functions.
    - num_knots: The number of knots to select.
    
    Returns:
    - A tensor containing the selected knots (padded).
    """
    def uniform_selection(num_knots,img_shape):
        """
        Creates a uniform knot vector for the given number of knots and image shape.
        """
        
        # number of pixels in a row
        num_pixels = img_shape[0]
        # divide into equal parts
        knot_vector = np.linspace(0,num_pixels-1,num_knots+2)
        #knot_vector = np.linspace(0,num_pixels,num_knots+2)
        # padding
        knot_vector = np.pad(knot_vector, (degree, degree), 'constant', constant_values=(min(knot_vector), max(knot_vector)))
        knot_vector = torch.tensor(knot_vector, dtype=torch.float32) 
        # return x and y knot vectors, which are the same for uniform
        return knot_vector,knot_vector
    
    def non_uniform_selection(num_knots,img_shape):
        """
        Creates a non-uniform knot vector for the given number of knots and image shape.
        """
        #find the variance of the dataset
        variance = torch.var(dataset.float(), dim=0)
        #find the indices of the top variance pixels
        
        sorted_indices = torch.argsort(variance.view(-1), descending=True)
        # Convert flattened indices to x, y indices
        x_indices = np.array(sorted_indices % img_shape[1])
        y_indices= np.array(sorted_indices // img_shape[1])
        #zip the indices together
        indices = list(zip(x_indices,y_indices))
        # select the top variance pixels but if x or y index exists already, skip it
        x_knots = []
        y_knots = []
        for x,y in indices: 
            if x not in x_knots and y not in y_knots and x != 0 and y != 0:
                x_knots.append(x)
                y_knots.append(y)
            if len(x_knots) == num_knots:
                break
        # sort the knots
        x_knots.sort()
        # add 0 and 27 to the beginning and end of the knot vector and pad
        x_knots=np.insert(x_knots,0,0)
        x_knots=np.append(x_knots,img_shape[0]-1)
        x_knots = np.pad(x_knots, (degree, degree), 'constant', constant_values=(min(x_knots), max(x_knots)))
        x_knots = torch.tensor(x_knots, dtype=torch.float32)

        y_knots.sort()
        y_knots=np.insert(y_knots,0,0)
        y_knots=np.append(y_knots,img_shape[1]-1)        
        y_knots = np.pad(y_knots, (degree, degree), 'constant', constant_values=(min(y_knots), max(y_knots)))
        y_knots = torch.tensor(y_knots, dtype=torch.float32)
        # return x and y knot vectors
        return x_knots,y_knots
    
    def fit_surface(images,test_set,degree,x_knots,y_knots,img_shape):
        """
        Fits a OB-spline surface to the dataset.
        """
        L1 = len(x_knots) - (2*degree) -1 
        L2 = len(y_knots) - (2*degree) -1

        # create the B-spline basis matrices
        A = B_spline_matrix(torch.arange(img_shape[0]), degree + 1 + L1 -1, degree, x_knots)
        B = B_spline_matrix(torch.arange(img_shape[1]), degree + 1 + L2 -1, degree, y_knots)
        # make them orthogonal
        A = gram_schmidt(A)
        B = gram_schmidt(B)
        # fit the surface
        # solve the least squares problem by calculating C = (A^T A)^-1 A^T G B (B^T B)^-1 where G is the image    
        # C = np.linalg.inv(A.T @ A) @ A.T @ image @ B @ np.linalg.inv(B.T @ B) 
        # don't need to find the inverses, as A^T A and B^TB are identity matrices given the orthonormality of the basis functions
        
        #training set        
        AG = torch.matmul(torch.transpose(A, 0, 1), images)
        AGB = torch.matmul(AG, B)
        C = AGB
        z_splines_train = torch.matmul(A, C) @ torch.transpose(B, 0, 1)  
        MSE_train = torch.mean((images - z_splines_train) ** 2) 

        #test set
        AG = torch.matmul(torch.transpose(A, 0, 1), test_set)
        AGB = torch.matmul(AG, B)
        C = AGB
        z_splines_test = torch.matmul(A, C) @ torch.transpose(B, 0, 1)
        MSE_test = torch.mean((test_set - z_splines_test) ** 2)
        return MSE_train, MSE_test, z_splines_train, z_splines_test
        
       
    order = degree +1
    #remove singleton dimension
    train_set = train_set.squeeze().float()
    img_shape = train_set.shape[1:]
    # let's just use the first image for debugging
    #train_set = train_set[0]
    # knot range to try
    knot_range = np.arange(4,20,2) #end is exclusive  
    #knot_range = [28] 
    # if uniform is true, use uniform knot spacing
    for num_knots in knot_range:
        if uniform:
            x_knots,y_knots = uniform_selection(num_knots,img_shape)
            #fit the surface
            MSE_train, MSE_test, z_splines_train, z_splines_test= fit_surface(train_set,test_set,degree,x_knots,y_knots,img_shape)
            print(f"MSE for uniform knot spacing with {num_knots} internal knots:")
            print("MSE for training set:")
            print(MSE_train.item())
            print("MSE for test set:")
            print(MSE_test.item())

        else:
            x_knots,y_knots = non_uniform_selection(num_knots,img_shape)
            #print(x_knots)
            #print(y_knots)
            #fit the surface
            MSE_train, MSE_test, z_splines_train, z_splines_test = fit_surface(train_set,test_set,degree,x_knots,y_knots,img_shape)
            print(f"MSE for non-uniform knot spacing with {num_knots} internal knots:")
            print("MSE for training set:")
            print(MSE_train.item())
            print("MSE for test set:")
            print(MSE_test.item())
  

if __name__ == "__main__":
    # load the datasets
    train_set, test_set = createDataSet(1)
    # scale test set to be between 0 and 1
    test_set = test_set.data.squeeze().float()/255 

    images_org = train_set.data
    samples_flow_1 = torch.load("sampled_images/flow_1/samples_flow_1.pt")
    samples_flow_2 = torch.load("sampled_images/flow_2/samples_flow_2.pt")
    # add to list
    datasets = [images_org, samples_flow_1, samples_flow_2]
    datasets = [images_org]
    for dataset in datasets:
        #scale to be between 0 and 1
        training_set = dataset.squeeze().float()/255 
        select_knots(training_set,test_set,3,uniform=True)
        select_knots(training_set,test_set,3,uniform=False)
