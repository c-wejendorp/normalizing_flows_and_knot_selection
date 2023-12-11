import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F

"""
This code is heavily inspired by the existing pytorch package called normflows: https://arxiv.org/abs/2302.12014
I did 'implement' it myself to obtain a richer understanding of the underlyinig concepts behind normalizing flows
Furthermore, it allowed me to finetune the implementation to the specific use case of my project
"""

class nfModel(nn.Module):
    def __init__(self,flows, base_distribution_type = "normal"):
        """
        Constructor for the nfModel class
        Arguments:
            flows (list): list of flows of type nn.Module. Should be ordered in the direction target distribution -> base distribution 
            base_distribution_type (str): type of base distribution. Currently, standard uniform, standard normal and standard_logistic are supported
            distribution_parameters (list): if uniform distribution, then [min,max], if normal distribution, then [mean,std]
        """        
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.base_distribution_type = base_distribution_type
        self.base_distribution,self.distribution_param = self.createBaseDistribution(base_distribution_type)        

    def createBaseDistribution(self,base_distribution_type):           
        if base_distribution_type == "uniform":
            base_distribution = distributions.uniform.Uniform(0,1)
        elif base_distribution_type == "normal":
            base_distribution = distributions.normal.Normal(0,1)
        elif base_distribution_type == "logistic":
            base_distribution = StandardLogistic()
        else:
            raise ValueError("Base distribution type not supported")
        return base_distribution, [0,1]
    
    def sampleFromBaseDistribution(self,samplesShape):  
        """
        Sample from the base distribution
        Arguments:
            samplesShape (shape): shape of the samples to be generated
        """    
        return self.base_distribution.sample(samplesShape)
    
    def normalizingDirection(self,x, return_log_det = False):
        """   
        Flow from the target distribution to the base distribution.          
        This is the 'normalizing' direction of the model and noted as u = f(x) in my project.
        Arguments:
            x (tensor): input tensor
        """         

        log_det = torch.zeros(len(x)) # initialize log_det to zero
        for flow in self.flows:
            x,_log_det = flow.normalizingDirection(x,return_log_det = True)        
            log_det += _log_det

        u = x # since we should now have transformed x to u
        if return_log_det:
            return u,log_det
        else:
            return u
        
    def generativeDirection(self,u,return_log_det = False):
        """   
        Flow from the base distribution to the target distribution.          
        This is the 'generative' direction of the model and noted as x = g(u) in my project.
        Arguments:
            u (tensor): input tensor
        """  
        log_det = torch.zeros(len(u)) # initialize log_det to zero

        # note that we reverse the order of the flows, as they are ordered from from target distribution to base distribution
        for flow in reversed(self.flows):
            u,_log_det = flow.generativeDirection(u,return_log_det = True)
            log_det += _log_det

        x = u # since we should now have transformed u to x
        if return_log_det:
            return x,log_det
        else:
            return x                 
            
    def forwardKL(self,x):
        """   
        Forward KL divergence between trained distribution and true distribution.
        Equivalent to the negative log-likelihood of the data given the model.

        Arguments:
            x (tensor): batch from the dataset
        """         
        # negative log likelihood  =  -1 / N * sum( log p_u(f(x)) + log | det J_f(x) | )
        # N is the number of samples in the batch        

        u,log_det = self.normalizingDirection(x,return_log_det = True) 

        if self.base_distribution_type == "uniform":

            # Create a mask for values outside the valid range
            mask = (u < self.distribution_param[0]) | (u > self.distribution_param[1])
            # clamp values and find the probability density            
            clamped  = torch.clamp(u,self.distribution_param[0],self.distribution_param[1])
            clamped_log_density = self.base_distribution.log_prob(clamped)
            # set the probability density to 0 for values outside the valid range
            u = clamped_log_density.where(~mask, torch.tensor(0.0))
            log_prob = u.view(len(x),-1).sum(1) # sum over the dimensions of the samples  

        else:  # we dont have a problem with the range for the normal and logistic distribution
            # since they are defined on the entire real line
            log_prob = self.base_distribution.log_prob(u).view(len(x),-1).sum(1) # sum over the dimensions of the samples      

            #eps = 1e-6        
            #log_prob = self.base_distribution.log_prob(u + eps ).view(len(x),-1).sum(1) # lets try add small valueo to avoid log(0) errors  

        negative_log_likelihood = -torch.mean(log_prob + log_det)

        return negative_log_likelihood


class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super(StandardLogistic, self).__init__()
        # the standard logistic distribution is defined on the entire real line as 
        # p(u) = exp(-u) / ((1 + exp(-u))^2)

    def log_prob(self, u):
        """Computes data log-likelihood.
        Args:
            u: input tensor.
        Returns:
            log-likelihood.
        """
        return -F.softplus(u) - F.softplus(-u) # as in  NICE.
        #return - u - 2. * F.softplus(-u) #  -u- 2 * log(1 + exp(-u)) equivalent to the above
    
    def sample(self, size):
        """Samples from the distribution.

        Args:
            size: number of samples to generate.
        Returns:
            samples.
        """
        z = torch.distributions.Uniform(0., 1.).sample(size)       
        return torch.log(z) - torch.log(1. - z)

class flow(nn.Module):
    def __init__(self):
        super().__init__()
        
    def generativeDirection(self,z_k,return_log_det = False):
        raise NotImplementedError
        
    def normalizingDirection(self,z_k,return_log_det = False):
        raise NotImplementedError




       
        
    



        
    
        
    
       
    

       
    
    

    
    

    




