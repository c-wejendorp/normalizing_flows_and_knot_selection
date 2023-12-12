import torch
import torch.nn as nn
from split_and_merge import *
from nfModel_class import *

class MLP(nn.Module):
    ''' 
    Creates the MLP used in the coupling layer
    Arguments:
        input_shape: tuple, shape of the input WITHOUT batch size
        num_hidden_layers: int, number of hidden layers
        num_hidden_units: int, number of hidden units in hidden each layer
        init_zeros: boolean if true, weights and biases of last layer are initialized with zeros
    '''
    def __init__(self,input_shape,num_hidden_layers,num_hidden_units,init_zeros=True,output_activation=None):
        super().__init__()            

        self.input_dim = torch.prod(torch.tensor(input_shape)) // 2 # divide by 2 since we split the input in two equal parts
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.MLP = self.createMLP(self.input_dim,self.num_hidden_layers,self.num_hidden_units,init_zeros=init_zeros)
        
        #self.MLP = self.createMLP(self.input_dim,self.num_hidden_layers,self.num_hidden_units,init_zeros=init_zeros)

    def createMLP(self,input_dim,num_hidden_layers,num_hidden_units,init_zeros=True,output_activation=None):
        """
        Creates the MLP used in the coupling layer
        Arguments:
            input_dim: int, dimension of the input and thus also the output
            num_hidden_layers: int, number of hidden layers
            num_hidden_units: int, number of hidden units in hidden each layer
            init_zeros: boolean if true, weights and biases of last layer are initialized with zeros
        """
        MLP = nn.ModuleList([])
        linear_layer_first = nn.Linear(input_dim, num_hidden_units)        
        MLP.append(linear_layer_first)
        MLP.append(nn.ReLU())

        for _ in range(num_hidden_layers -2 ):
            linear_layer = nn.Linear(num_hidden_units, num_hidden_units)           
            
            MLP.append(linear_layer)
            MLP.append(nn.ReLU())           
          
        linear_layer_last = nn.Linear(num_hidden_units, input_dim)
        if init_zeros:
            nn.init.zeros_(linear_layer_last.weight)
            nn.init.zeros_(linear_layer_last.bias)           
            
        MLP.append(linear_layer_last)
        # batch norm
        #MLP.append(nn.BatchNorm1d(input_dim))

        if output_activation is not None:
            if output_activation == "sigmoid":
                MLP.append(nn.Sigmoid())
            elif output_activation == "tanh":
                MLP.append(nn.Tanh())
            elif output_activation == "relu":
                MLP.append(nn.ReLU())
            else:
                raise ValueError("output_activation must be either sigmoid, tanh or relu.")        
        
        # turn MLP into a sequential model
        MLP = nn.Sequential(*MLP)
        return MLP     
    
    def forward(self,x):
        """
        Arguments:
            x: tensor of shape (batch_size,channels,height,width)
        Returns:
            tensor of shape (batch_size,channels,height,width)
        """
        return self.MLP(x)     
        

class couplingLayer(flow):
    def __init__(self,input_shape,num_hidden_layers=5,num_hidden_units=1000,scale=True,shift=True,init_zeros=True,output_activation=None):
        """
        Layer that implements the coupling layer: z1 = z1, z2 = z2 * NN_scale(z1) + NN_shift(z1)
        Arguments: 
            input_shape: tuple, shape of the input WITHOUT batch size. It is BEFORE the split!

            num_hidden_layers: int, number of hidden layers in the MLPs
            num_hidden_units: int, number of hidden units in hidden each layer in the MLPs

            scale: boolean, if True, a scale parameter is learned
            shift: boolean, if True, a shift parameter is learned
        """      
        super().__init__()
        self.input_shape = input_shape
        # check if either scale or shift is True
        assert scale or shift, "At least one of scale and shift must be True."

        if scale:
            self.add_module("scale",MLP(input_shape,num_hidden_layers,num_hidden_units,init_zeros=init_zeros,output_activation=output_activation))
        if shift:
            self.add_module("shift",MLP(input_shape,num_hidden_layers,num_hidden_units,init_zeros=init_zeros,output_activation=output_activation))

    def normalizingDirection(self,zs : list ,return_log_det=False):
        """
        Arguments: 
            zs: list [z1,z2] of tensors

        Returns:
            list [z1,z2] of tensors. z1 is unchanged, z2 is transformed by the MLPs
        """
        z1,z2 = zs

        #shape pre flatten
        z1_org_shape = z1.shape
        z2_org_shape = z2.shape

        # flatten 
        #z1 = z1.view(z1.shape[0], -1)
        #z2 = z2.view(z2.shape[0], -1)

        # flatten with reshape
        z1 = z1.reshape(z1.shape[0], -1)
        z2 = z2.reshape(z2.shape[0], -1)


        # pass through MLPs and keep track of log determinant              
        log_det = torch.zeros(z2.shape[0])        

        if hasattr(self,"scale"):
            scale = self.scale(z1)
            # use the exponential function to ensure positivity
            # also makes the log determinant calculation easier
            z2 = z2 * torch.exp(scale)            
            log_det += torch.sum(scale) 

        if hasattr(self,"shift"):
            shift = self.shift(z1)
            z2 = z2 + shift            
            # shift does not change the log determinant
        
        # now unflatten
        #z1 = z1.view(z1.shape[0],self.input_shape[0],self.input_shape[2],-1)
        #z2 = z2.view(z2.shape[0],self.input_shape[0],self.input_shape[2],-1)
        z1 = z1.view(z1_org_shape)
        z2 = z2.view(z2_org_shape)
        

        if return_log_det:
            return [z1,z2], log_det
        else:
            return [z1,z2]

    def generativeDirection(self, zs : list , return_log_det=False):
        """
        Arguments: 
            zs: list [z1,z2] of tensors

        Returns:
            list [z1,z2] of tensors. z1 is unchanged, z2 is transformed by the MLPs
        """
        z1,z2 = zs

        #shape pre flatten
        z1_org_shape = z1.shape
        z2_org_shape = z2.shape

        # flatten 
        # z1 = z1.view(z1.shape[0], -1)
        # z2 = z2.view(z2.shape[0], -1)

        # flatten with reshape
        z1 = z1.reshape(z1.shape[0], -1)
        z2 = z2.reshape(z2.shape[0], -1)

        # pass through MLPs and keep track of log determinant              
        log_det = torch.zeros(z2.shape[0])       

        # NOTE remember now the order of the operation is reversed, 
        # so we need to reverse the order of the operations 
        # compared to the normalizing direction

        if hasattr(self,"shift"):
            shift = self.shift(z1)
            z2 = z2 - shift
            # shift does not change the log determinant
            

        if hasattr(self,"scale"):
            scale = self.scale(z1)
            # use the exponential function to ensure positivity
            # also makes the log determinant calculation easier
            z2 = z2 / torch.exp(scale)             
            log_det += -torch.sum(scale) # minus sign since we divide by the scale        
        
        # now unflatten
        # z1 = z1.view(z1.shape[0],self.input_shape[0],self.input_shape[2],-1)
        # z2 = z2.view(z2.shape[0],self.input_shape[0],self.input_shape[2],-1)
        z1 = z1.view(z1_org_shape)
        z2 = z2.view(z2_org_shape)

        if return_log_det:
            return [z1,z2], log_det
        else:
            return [z1,z2]       
        
class scaleAll(flow):
    def __init__(self,input_shape,preprocess=False):
        """
        Layer that scales all dimensions of the input
        Arguments: 
            input_shape: tuple, shape of the input WITHOUT batch size. It is BEFORE the split!
            preprocess: boolean, if True, this layer is used to scale the input to [-1,1]
        """      
        super().__init__()
        self.input_shape = input_shape
        self.preprocess = preprocess
        # self.scale = nn.Parameter(torch.zeros(input_shape)) # cannot be zeroes since we divide by this
        if preprocess:
            self.scale = torch.ones(input_shape) / 128.0 
        else:
            # initialize scaling parameters randomly
            self.scale = nn.Parameter(torch.rand(input_shape))
            #self.scale = nn.Parameter(torch.ones(input_shape))

    def normalizingDirection(self,z,return_log_det=False):
        if self.preprocess:
            z = (self.scale * z) - 1.0 # scale to [-1,1]
            log_det = torch.sum(torch.log(self.scale)) 
        else:            
            z = z * torch.exp(self.scale)            
            log_det = torch.sum(self.scale)
        
        if return_log_det:
            return z, log_det
        else:
            return z
    
    def generativeDirection(self,z,return_log_det=False):
        if self.preprocess:
            z = (z + 1.0) / self.scale # scale to [0,256]
            log_det = -torch.sum(torch.log(self.scale))
        else:
            z = z / torch.exp(self.scale)
            log_det = -torch.sum(self.scale)
        if return_log_det:
            return z, log_det
        else:
            return z 
        
class linearFlow(flow):
    def __init__(self,input_shape):
        """
        """            
        super().__init__()
        self.input_shape = input_shape
        self.W = nn.Parameter(self.sampleOrthogonalMatrix(input_shape))

    def sampleOrthogonalMatrix(self,input_shape):
        """
        Samples an orthogonal matrix of with shape (c,h,w)        
        """
        # sample random matrix
        W = torch.randn(input_shape)
        # compute QR decomposition
        Q, _ = torch.torch.linalg.qr(W)
        # compute determinant
        #det = torch.det(Q)
        ## if determinant is negative, flip the sign of the first column
        #if det < 0:
        #    Q[:,0] = -Q[:,0]
        return Q

    def normalizingDirection(self,z,return_log_det=False):
        z = torch.matmul(self.W,z)
        log_det = torch.log(torch.abs(torch.det(self.W)))
        if return_log_det:
            return z, log_det
        else:
            return z
        
    def generativeDirection(self,z,return_log_det=False):
        z = torch.matmul(torch.inverse(self.W),z)
        log_det = torch.log(torch.abs(torch.det(torch.inverse(self.W))))
        if return_log_det:
            return z, log_det
        else:
            return z        

if __name__ == "__main__":
    # create a random input with values in range [0,256]
    x = torch.rand((1,1,4,4)) * 256
    # preprocessing layer
    preprocess = scaleAll(x.shape[1:],preprocess=True)   
    # create a coupling block, which is split, coupling layer and merge
    split = Split(split_type="columns")
    coupling_layer = couplingLayer(input_shape=x.shape[1:])
    merge = Merge(split_type="columns")
    # scaling layer
    scale = scaleAll(x.shape[1:],preprocess=False)
    # linear layer
    linear = linearFlow(x.shape[1:])


    #now lets go through the normalizing direction
    z = preprocess.normalizingDirection(x)

    z = linear.normalizingDirection(z)
    
    # split the input
    z1,z2 = split.normalizingDirection(z)
    # apply the coupling layer
    z1,z2 = coupling_layer.normalizingDirection([z1,z2])
    # merge the input    
    # check that the shape is the same
    x_latent = merge.normalizingDirection([z1,z2])
    assert x.shape == x_latent.shape, "Shapes are not the same after normalizing direction."

    # check that the generative direction is the inverse of the normalizing direction
    z1,z2 = merge.generativeDirection(x_latent)
    z1,z2 = coupling_layer.generativeDirection([z1,z2])
    x_recon = split.generativeDirection([z1,z2])
    x_recon = linear.generativeDirection(x_recon)
    x_recon = preprocess.generativeDirection(x_recon)

    assert torch.allclose(x,x_recon), "Generative direction is not the inverse of the normalizing direction."
    print("The coupling layer is invertable.")


   



    
