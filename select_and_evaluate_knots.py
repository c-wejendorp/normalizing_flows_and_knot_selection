import numpy as np
import torch
import matplotlib.pyplot as plt
from data_functions import *
import os
from spline_functions import *
import pandas as pd


def uniform_selection(num_knots,img_shape,degree):
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

def non_uniform_selection(train_set,num_knots,img_shape,degree):
    """
    Creates a non-uniform knot vector for the given number of knots and image shape.
    """
    #find the variance of the dataset
    variance = torch.var(train_set.float(), dim=0)
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
    non_allowed_indices = [0,img_shape[0]-1]
    for x,y in indices: 
        if x not in x_knots and not x in non_allowed_indices:
            x_knots.append(x)
        if y not in y_knots and not x in non_allowed_indices:
            y_knots.append(y)
        if len(x_knots) == num_knots and len(y_knots) == num_knots:
            break
    # if there to many indices, remove the last ones        
    x_knots = x_knots[:num_knots]        
    y_knots = y_knots[:num_knots]

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





def select_knots(train_set,test_set,degree,uniform=False,scale=None):
    """
    Selects the knots for the B-spline basis functions using the k-means algorithm.
    
    Parameters:
    - dataset: The dataset to use for the k-means algorithm.
    - B-spline_order: The order of the B-spline basis functions.
    - num_knots: The number of knots to select.
    
    Returns:
    - A tensor containing the selected knots (padded).
    """   
        
    def fit_surface(images,test_set,degree,x_knots,y_knots,img_shape,scale=None):
        """
        Fits a OB-spline surface to the dataset.
        """
        L1 = len(x_knots) - (2*degree) -1 
        L2 = len(y_knots) - (2*degree) -1

        
        # create the B-spline basis matrices
        A = B_spline_matrix(torch.arange(img_shape[0]), degree + 1 + L1 -1, degree, x_knots)
        B = B_spline_matrix(torch.arange(img_shape[1]), degree + 1 + L2 -1, degree, y_knots)
        A = gram_schmidt(A,use_QR=False)
        B = gram_schmidt(B,use_QR=False)

        if scale is not None:
            A_scaled = B_spline_matrix(torch.linspace(0,img_shape[0],img_shape[0]*scale), degree + 1 + L1 -1, degree, x_knots)
            B_scaled = B_spline_matrix(torch.linspace(0,img_shape[1],img_shape[1]*scale), degree + 1 + L2 -1, degree, y_knots)
            A_scaled = gram_schmidt(A_scaled,use_QR=False)
            B_scaled = gram_schmidt(B_scaled,use_QR=False)                

        # fit the surface
        # solve the least squares problem by calculating C = (A^T A)^-1 A^T G B (B^T B)^-1 where G is the image    
        # C = np.linalg.inv(A.T @ A) @ A.T @ image @ B @ np.linalg.inv(B.T @ B) 
        # don't need to find the inverses, as A^T A and B^TB are identity matrices given the orthonormality of the basis functions
                
        #training set        
        AG = torch.matmul(torch.transpose(A, 0, 1), images)
        AGB = torch.matmul(AG, B)
        C = AGB
        
        if scale is None:
            z_splines_train = torch.matmul(A, C) @ torch.transpose(B, 0, 1)            
            MSE_train = torch.mean((images - z_splines_train) ** 2)
        else:
            z_splines_train = torch.matmul(A_scaled, C) @ torch.transpose(B_scaled, 0, 1)
            MSE_train = None

        #test set
        AG = torch.matmul(torch.transpose(A, 0, 1), test_set)
        AGB = torch.matmul(AG, B)
        C = AGB        
        if scale is None:
            z_splines_test = torch.matmul(A, C) @ torch.transpose(B, 0, 1)
            MSE_test = torch.mean((test_set - z_splines_test) ** 2)
        else:
            z_splines_test = torch.matmul(A_scaled, C) @ torch.transpose(B_scaled, 0, 1)
            MSE_test = None

        return MSE_train, MSE_test, z_splines_train, z_splines_test
        
       
    order = degree +1
    #remove singleton dimension
    train_set = train_set.squeeze().float()
    img_shape = train_set.shape[1:]
    # let's just use the first image for debugging
    #train_set = train_set[0]
    # knot range to try
    knot_range = np.arange(2,24,2) #end is exclusive  
    #knot_range = [28] 
    # if uniform is true, use uniform knot spacing
    # Initialize an empty DataFrame
    result_df = pd.DataFrame(columns=['Type', 'Internal Knots', 'MSE Train', 'MSE Test'])
    
    test_z_splines = []

    for num_knots in knot_range:
        if uniform:
            x_knots, y_knots = uniform_selection(num_knots, img_shape, degree)
        else:
            x_knots, y_knots = non_uniform_selection(train_set,num_knots, img_shape, degree)

        # Fit the surface
        MSE_train, MSE_test, z_splines_train, z_splines_test = fit_surface(train_set, test_set, degree, x_knots, y_knots, img_shape,scale=scale)
        test_z_splines.append(z_splines_test)


        # Determine the type of knot spacing
        knot_spacing_type = 'Uniform' if uniform else 'Non-Uniform'

        if scale is None:
            # Add results to the DataFrame
            result_df.loc[len(result_df)] = [knot_spacing_type, num_knots, round(MSE_train.item(),4), round(MSE_test.item(),4)]
        


    # Display the results DataFrame
    print(result_df)
    return result_df, test_z_splines
  

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
    names = ["Original", "Flow_1", "Flow_2"]

    #if we do scaling
    scale = 4 
    
    #datasets = [images_org]
    for idx,dataset in enumerate(datasets):
        #scale to be between 0 and 1
        training_set = dataset.squeeze().float()/255
        print(names[idx])
        #select the knots        
        df_uniform,test_z_splines_uniform=select_knots(training_set,test_set,3,uniform=True)
        df_non_uniform,test_z_splines_list_nonuniform=select_knots(training_set,test_set,3,uniform=False)
        #save the dataframes
        df_uniform.to_csv("dataframes/df_uniform_"+names[idx]+".csv")
        df_non_uniform.to_csv("dataframes/df_non_uniform_"+names[idx]+".csv")
        #save the test z splines
        torch.save(test_z_splines_uniform,"OB_surfaces/test_OB_surfaces_uniform_"+names[idx]+".pt")
        torch.save(test_z_splines_list_nonuniform,"OB_surfaces/test_OB_surfaces_non_uniform_"+names[idx]+".pt")

        #now with scale
        print(names[idx]+" with scale")
        #select the knots
        df_uniform,test_z_splines_uniform=select_knots(training_set,test_set,3,uniform=True,scale=scale)
        df_non_uniform,test_z_splines_list_nonuniform=select_knots(training_set,test_set,3,uniform=False,scale=scale)
        # the dataframes are not interesting in this case
        #save the test z splines which are scaled
        torch.save(test_z_splines_uniform,"OB_surfaces/test_OB_surfaces_uniform_"+names[idx]+f"_scale_{scale}.pt")
        torch.save(test_z_splines_list_nonuniform,"OB_surfaces/test_OB_surfaces_non_uniform_"+names[idx]+f"_scale_{scale}.pt")

