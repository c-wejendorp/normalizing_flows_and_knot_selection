import torch
import torch.nn as nn

from nfModel_class import *

class Split(flow):
    def __init__(self,split_type="columns",input_shape=None):
        """
        Arguments: 
            split_type: columns, columns_odd, rows, rows_odd, checkerboard, checkerboard_odd, top_bottom, bottom_top.
            input_shape: shape of the input tensor without batchsize. Only needed if split_type is random.        
        """     
        super().__init__()
        self.split_type = split_type
        self.perm = None
        if split_type == "random":
            if input_shape is None:
                raise ValueError("input_shape must be specified if split_type is random.")
            # we need to keep track of the permutation used for the random split
            self.perm = torch.randperm(input_shape[0]*input_shape[1]*input_shape[2]) 
                     
               
    def normalizingDirection(self,z_k : torch.tensor ,return_log_det=False):
        """
        Arguments: 
            z_k: tensor of shape (batch_size,channels,height,width)

        Returns:
            list [z1,z2] of tensors. If split_type is columns, z1 is all the even columns and z2 is all the odd columns.
            If the split_type is columns_odd then z1 is all the odd columns and z2 is all the even columns.
        """
        if self.split_type == "columns":
            z1 = z_k[:,:,:,::2]
            z2 = z_k[:,:,:,1::2]
        elif self.split_type == "columns_odd":
            z1 = z_k[:,:,:,1::2]
            z2 = z_k[:,:,:,::2]
        elif self.split_type == "rows":
            z1 = z_k[:,:,::2,:]
            z2 = z_k[:,:,1::2,:]
        elif self.split_type == "rows_odd":
            z1 = z_k[:,:,1::2,:]
            z2 = z_k[:,:,::2,:]
        elif self.split_type == "checkerboard":
            z1 = torch.zeros_like(z_k[:,:,:,::2])
            z2 = torch.zeros_like(z_k[:,:,:,1::2])

            for i in range(z_k.shape[3]):
                if i % 2 == 0:
                    z1[:,:,i,:] = z_k[:,:,i,::2] 
                    z2[:,:,i,:] = z_k[:,:,i,1::2] 
                else:
                    z1[:,:,i,:] = z_k[:,:,i,1::2]
                    z2[:,:,i,:] = z_k[:,:,i,::2]              
           
        elif self.split_type == "checkerboard_odd":
            z1 = torch.zeros_like(z_k[:,:,:,::2])
            z2 = torch.zeros_like(z_k[:,:,:,1::2])

            for i in range(z_k.shape[3]):
                if i % 2 == 0:
                    z1[:,:,i,:] = z_k[:,:,i,1::2] 
                    z2[:,:,i,:] = z_k[:,:,i,::2] 
                else:
                    z1[:,:,i,:] = z_k[:,:,i,::2]
                    z2[:,:,i,:] = z_k[:,:,i,1::2]
        elif self.split_type == "top_bottom":
            z1 = z_k[:,:,:z_k.shape[2]//2,:]
            z2 = z_k[:,:,z_k.shape[2]//2:,:]
        elif self.split_type == "bottom_top":
            z1 = z_k[:,:,z_k.shape[2]//2:,:]
            z2 = z_k[:,:,:z_k.shape[2]//2,:]

        elif self.split_type == "random":
            # flatten the last two dimensions
            z_temp = torch.flatten(z_k, start_dim=1)                               
                        
            # permute the vector
            z_temp = z_temp[:,self.perm]
            # reshape back to original shape
            z_temp = z_temp.reshape(z_k.shape)
            # split in two
            z1 = z_temp[:,:,:z_k.shape[2]//2,:]
            z2 = z_temp[:,:,z_k.shape[2]//2:,:] 
            #print([z1,z2])
               

        else:
            raise ValueError("split_type must be one of columns, columns_odd, rows, rows_odd, checkerboard and checkerboard_odd.")
        
        # we return a log determinant of zero, since no transformation of values is done
        if return_log_det:
            return [z1,z2], torch.zeros(z_k.shape[0]) # zeroes of shape (batch_size)
        else:
            return [z1,z2]
    
    def generativeDirection(self, z_list : list , return_log_det=False):
        """
        Arguments: 
            z_k: list [z1,z2] of tensors

        Returns:
            single tensor of shape (batch_size,channels,height,width)       
        """
        z1,z2 = z_list
        # check if they have same dimensions
        assert z1.shape == z2.shape

        if self.split_type == "columns":
            # now we merge the two halves back together, some times refered to as interleaving
            z_k = torch.stack((z1, z2), dim=-1)
            z_k = z_k.view(z1.shape[0], z1.shape[1], z1.shape[2], -1)

        elif self.split_type == "columns_odd":

            z_k = torch.stack((z2, z1), dim=-1)
            z_k = z_k.view(z1.shape[0], z1.shape[1], z1.shape[2], -1)
            
        elif self.split_type == "rows":
            z_k = torch.stack((z1, z2), dim=-2)
            z_k = z_k.view(z1.shape[0], z1.shape[1],-1,z1.shape[3])
            #raise ValueError("split_type_ not invertible yet.")
            
        elif self.split_type == "rows_odd":
            z_k = torch.stack((z2, z1), dim=-2)
            z_k = z_k.view(z1.shape[0], z1.shape[1],-1,z1.shape[3])
           
        elif self.split_type == "checkerboard":
            batch_dim = z1.shape[0]
            channel_dim = z1.shape[1]
            height_dim = z1.shape[2]
            width_dim = z1.shape[3] *2 # since we have two tensors of half the width
            z_k = torch.zeros((batch_dim,channel_dim,height_dim,width_dim))

            for i in range(height_dim):
                if i % 2 == 0:
                    z_k[:,:,i,::2] = z1[:,:,i,:]
                    z_k[:,:,i,1::2] = z2[:,:,i,:] 
                else:
                    z_k[:,:,i,::2] = z2[:,:,i,:]
                    z_k[:,:,i,1::2] = z1[:,:,i,:]           
            
        elif self.split_type == "checkerboard_odd":
            batch_dim = z1.shape[0]
            channel_dim = z1.shape[1]
            height_dim = z1.shape[2]
            width_dim = z1.shape[3] *2 # since we have two tensors of half the width
            z_k = torch.zeros((batch_dim,channel_dim,height_dim,width_dim))

            for i in range(height_dim):
                if i % 2 == 0:
                    z_k[:,:,i,::2] = z2[:,:,i,:]
                    z_k[:,:,i,1::2] = z1[:,:,i,:] 
                else:
                    z_k[:,:,i,::2] = z1[:,:,i,:]
                    z_k[:,:,i,1::2] = z2[:,:,i,:]    
        elif self.split_type == "top_bottom":
            z_k = torch.cat((z1,z2),dim=2)
        elif self.split_type == "bottom_top":
            z_k = torch.cat((z2,z1),dim=2)

        elif self.split_type == "random":
            # merge the two tensors
            z_k = torch.cat((z1,z2),dim=2)
            # flatten the tensor
            z_temp = torch.flatten(z_k, start_dim=1)
            # permute the vector back
            inverse_perm = torch.arange(z_temp.shape[1])
            inverse_perm[self.perm] = torch.arange(z_temp.shape[1])
            z_temp = z_temp[:,inverse_perm]            
            # reshape back to original shape
            z_k = z_temp.reshape(z_k.shape)
            

        else:
            raise ValueError("split_type must be one of columns, columns_odd, rows, rows_odd, checkerboard and checkerboard_odd.")
        
        # we return a log determinant of zero, since no transformation of values is done
        if return_log_det:
            return z_k, torch.zeros(z_k.shape[0]) # zeroes of shape (batch_size)
        else:
            return z_k       
        
class Merge(Split):
    """
    Same as Split but with normalizing and generative pass interchanged
    """

    def __init__(self,split_type="columns",input_shape=None,permutation=None):        
            super().__init__(split_type,input_shape)
            # overwrite the random permutation that is created when the Split class is initialized
            # when the split_type is random
            self.perm = permutation
            

    def normalizingDirection(self, z, return_log_det=False):
        return super().generativeDirection(z,return_log_det=return_log_det)

    def generativeDirection(self,z,return_log_det=False):
        return super().normalizingDirection(z,return_log_det=return_log_det)

if __name__ == "__main__":
    # check if all types of splits are invertible
    #x = torch.randn(1,3,32,32)
    x = torch.randn(1,1,4,4)
    #for split_type in ["random"]:    
    for split_type in ["columns","columns_odd","rows","rows_odd","checkerboard","checkerboard_odd","top_bottom","bottom_top","random"]:
        split = Split(split_type=split_type,input_shape=x.shape[1:])
        merge = Merge(split_type=split_type,input_shape=x.shape[1:],permutation=split.perm)
        # merge = Merge(split_type=split_type)
        z_list,log_det = split.normalizingDirection(x,return_log_det=True)
        # x_recon,log_det = split.generativeDirection(z_list,return_log_det=True)
        x_recon,log_det = merge.normalizingDirection(z_list,return_log_det=True)
        assert torch.allclose(x,x_recon)
        print(f"Split type {split_type} is invertible.")
