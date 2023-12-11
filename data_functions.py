import torch
import torchvision
import matplotlib.pyplot as plt
import math

class Dequantize:
    """Adds uniform noise to each pixel of the image."""

    def __init__(self, min=0.0, max=255.0):              
        self.min = min
        self.max = max

    def __call__(self, x):
        """Scale back and add noise to each pixel."""
        #To tensor scales the data to [0,1], so we scale it back to [0,255]
        x = x * (self.max - self.min) + self.min
        x = x + torch.rand_like(x) # add uniform noise form [0,1)
        # data is now in range [0,256)        
        return x   

def createDataSet(digit):
    # transform to tensor in range [0,1] and add uniform noise
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),Dequantize()])   
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    trainset.data = trainset.data[trainset.targets == digit]
    trainset.targets = trainset.targets[trainset.targets == digit]
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms)
    testset.data = testset.data[testset.targets == digit]
    testset.targets = testset.targets[testset.targets == digit]
    return trainset, testset

# Define a function to plot images from a batch
def show_batch_images(images, labels, batch_size,clamp=False):    
    rows = int(math.sqrt(batch_size))
    cols = rows
    for i in range(batch_size):
        plt.subplot(rows, cols, i + 1)
        # change the first pixel to zero
        if clamp:
            #clamp values to [0,1]
            image_x = torch.clamp(images[i],0,1)
            plt.imshow(image_x[0].detach().numpy(), cmap='gray')
        else: 

            plt.imshow(images[i][0].detach().numpy(), cmap='gray')
        #plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Test the functions
    trainset, testset = createDataSet(1)
    # load some images from the training set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=25, shuffle=True)
    images, labels = next(iter(trainloader))
    # plot the images
    show_batch_images(images, labels, 25)

    