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
def show_batch_images(images, batch_size, clamp=False,save=True,model_name=None):
    rows = int(math.sqrt(batch_size))
    cols = rows

    if clamp:
        # Clamp values to [0, 255.9]
        images = torch.clamp(images, 0, 255.9)

    # Floor the values (corresponds to dequantization)
    images = torch.floor(images)

    fig, axs = plt.subplots(rows, cols, figsize=(rows, cols))
    
    for i in range(batch_size):
        ax = axs[i // rows, i % cols]

        ax.imshow(images[i][0].detach().numpy(), cmap='gray')
        # ax.set_title(f"Label: {labels[i].item()}")  # Uncomment if you have labels
        ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1) 
    # add title
    title = f"Samples from {model_name}, clamped" if clamp else f"Samples from {model_name}"
    plt.suptitle(title)
    #save fig
    plt.savefig(f"{title}.png", dpi=300) if save else plt.show()

if __name__ == "__main__":
    # Test the functions
    trainset, testset = createDataSet(1)
    # load some images from the training set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=25, shuffle=True)
    images, labels = next(iter(trainloader))
    # plot the images
    show_batch_images(images, labels, 25)

    