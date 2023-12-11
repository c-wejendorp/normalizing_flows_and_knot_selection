import torch
from nfModel_class import *
from coupling_layer import *
from split_and_merge import *
from data_functions import *
import numpy as np

# choose a digit to work with
digit = 1
# choose bath size
batch_size = 64
# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Create the dataset
# note that this adds uniform noises to the images and scales the pixel values to [0,1]
trainset, testset = createDataSet(digit)
# Create a DataLoader for the dataset
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# dimensions of the images
img_shape=trainset[0][0].shape

# Set up model
# Choose base distribution for the normalizing flow. Either uniform or Gaussian.
#base = "normal"
#distribution_param = [0.5,0.5/3] # min max if uniform and mean std if normal

base = "logistic" # works good

# base = "uniform" # doesnt work :(
# distribution_param = [0,1] # doesnt work :(

# add all the flows you want to use, remember to add split and merge flows before a coupling flow
# also remember to apply coupling to all pixels
flows = []

# here is an example which should be very similar to the NICE paper, however 
# it hard to really figure out how they split the input before for the additive coupling layer
# here it is just split by columns

# split_order = ["columns","columns_odd","rows","rows_odd","checkerboard","checkerboard_odd"]

# works decently
split_order = ["columns","columns_odd","columns","columns_odd"]
# lets try change the last bit to top_bottom and bottom_top
#split_order = ["top_bottom","bottom_top","top_bottom","bottom_top"]


# add the preprocessing flow
flows.append(scaleAll(img_shape,preprocess=True))

for type in split_order:
    flows.append(Split(split_type=type))
    # line below works, but not good results. requires normlayer added in coupling layer
    #flows.append(couplingLayer(img_shape,num_hidden_layers=5,num_hidden_units=1000,scale=True,init_zeros=True,output_activation="tanh"))
    flows.append(couplingLayer(img_shape,num_hidden_layers=5,num_hidden_units=1000,scale=False,init_zeros=True))
    flows.append(Merge(split_type=type))

# add a special type of flow that scale every dim similar to the NICE paper

flows.append(scaleAll(img_shape))

#flows.append(sigmoidAll())

# Construct flow model
model = nfModel(flows, base_distribution_type = base)


# check that the model is invertible
# Get the first batch of data
for images, labels in train_loader:
    break  # Break after the first batch

#lets try to transform some samples to the base distribution
number_of_images = 4 
images = images[:number_of_images,:,:,:]
labels = np.ones(number_of_images)

u_s = model.normalizingDirection(images) # normalizing direction
x_s = model.generativeDirection(u_s) # back again through the generative direction

print("Sanity check:")
#print(torch.isclose(images,x_s).all()) # should be true, but might not be due to numerical errors
print(torch.isclose(images, x_s, atol=1e-4, rtol=1e-8).all())

# now lets train the model

# Choose optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #lr=1e-3 is ok for normal base distribution


# the best results are obtained with standard logistic, and the following optimizer. No sigmoid layer. 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)

num_epochs = 1000
model.train()

loss_hist = []
for epoch in range(num_epochs):

    mean_loss_pr_batch = []
    
    #train_loss_batches = []
    
    for batch, _ in train_loader:
        #inputs, targets = inputs.to(device), targets.to(device)

        loss  = model.forwardKL(batch)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss_pr_batch.append(loss.detach().to('cpu').numpy())

    loss_hist.append(np.mean(mean_loss_pr_batch))
    print("Epoch: ", epoch, " Mean Loss pr batch: ", np.mean(mean_loss_pr_batch))           

print("Finished training.")

# now lets try some sampling
num_samples = 4
labels = np.ones(num_samples)
samples = model.sampleFromBaseDistribution([num_samples]+list(img_shape))
samples = model.generativeDirection(samples)
show_batch_images(samples, labels, num_samples)

# save the model and flow structure

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'flows': flows,
    'lost_hist': loss_hist
}

torch.save(checkpoint, f'NICE_logistic-top_bot_split_types-seed_{seed}_1000_epochs.pth')




















