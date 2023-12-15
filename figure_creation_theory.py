import numpy as np
import torch
import matplotlib.pyplot as plt
from spline_functions import *
import os
from scipy.stats import logistic

def plot_Bsplines(max_degree):
    max_order = max_degree + 1
    # Knot vectors
    knot_vector_np = np.arange(0, max_order + 3)
    t_values = np.linspace(min(knot_vector_np), max(knot_vector_np), 1000)

    # Turn into torch tensors
    knot_vector = torch.tensor(knot_vector_np, dtype=torch.float32)
    t_values = torch.tensor(t_values, dtype=torch.float32)

    # Create subplots
    fig, axs = plt.subplots(max_degree + 1, 2, figsize=(10, 2 * (max_degree + 1)))

    # Iterate over degrees
    for degree in range(max_degree + 1):
        # Pad the knot vector
        knot_vector_pad = np.pad(knot_vector_np, (degree, degree), 'constant', constant_values=(min(knot_vector_np), max(knot_vector_np)))
        knot_vector_pad = torch.tensor(knot_vector_pad, dtype=torch.float32)

        basis_values = torch.zeros((len(t_values), len(knot_vector) - degree - 1))
        basis_values_pad = torch.zeros((len(t_values), len(knot_vector_pad) - degree - 1))

        for i in range(len(knot_vector) - degree - 1):
            basis_values[:, i] = B_spline_torch(t_values, i, degree, knot_vector)
            axs[degree, 0].plot(t_values.numpy(), basis_values[:, i], label=f'Basis {i}')
        for i in range(len(knot_vector_pad) - degree - 1):    
            basis_values_pad[:, i] = B_spline_torch(t_values, i, degree, knot_vector_pad)            
            axs[degree, 1].plot(t_values.numpy(), basis_values_pad[:, i], label=f'Basis {i} padded knot vector')
        # add title and labels
        # add dashed lines to indicate knots
        for knot in knot_vector:
            axs[degree, 0].axvline(knot, linestyle='--', color='k',alpha=0.5)
            axs[degree, 1].axvline(knot, linestyle='--', color='k',alpha=0.5)
        
        axs[degree, 0].set_title(f'B-splines of Degree {degree}')
        axs[degree, 1].set_title(f'B-splines of Degree {degree}, padded knot vector')   
    plt.tight_layout()
    # save figure
    plt.savefig(os.path.join('figures', 'Bsplines.png'))
    plt.close()

def plot_single_Bplines(degree,uniform=True,orthogonal=False):
    if uniform:
        knot_vector = np.arange(0, 6+1)
    else:    
        knot_vector = np.array([0, 0.5, 1, 4, 4.5, 5, 6])
    # pad knot vector
    knot_vector = np.pad(knot_vector, (degree, degree), 'constant', constant_values=(min(knot_vector), max(knot_vector)))
    knot_vector = torch.tensor(knot_vector, dtype=torch.float32)

    t_values = torch.linspace(min(knot_vector), max(knot_vector), 1000)

    # Plot each basis function

    B_splines = B_spline_matrix(t_values, len(knot_vector) - degree - 1, degree, knot_vector)
    # check rank of B_splines
    print(torch.linalg.matrix_rank(B_splines))
    print(B_splines.shape[1])
    # check that it has full rank
    #assert torch.linalg.matrix_rank(B_splines) == B_splines.shape[1]
    if orthogonal:
        #B_splines = torch.linalg.qr(B_splines)[0]
        B_splines = gram_schmidt(B_splines)
    else:
        pass
    for column in range(B_splines.shape[1]):
        plt.plot(t_values.numpy(), B_splines[:, column], label=f'B-spline {column+1}')

    #for i in range(len(knot_vector) - degree - 1):
        #basis_values = [B_spline_naive(x, i, degree, knot_vector) for x in t_values]
        # use optimized version
    #    basis_values = B_spline_optimized(t_values, i, degree, knot_vector)      


    #    plt.plot(t_values.numpy(), basis_values, label=f'B-spline {i+1}')
    #add dashed lines to indicate knots
    for knot in knot_vector:
        plt.axvline(knot, linestyle='--', color='k',alpha=0.5)

    title = f'OB-splines of Degree {degree}' if orthogonal else f'B-splines of Degree {degree}'
    title += ', Uniform Knots' if uniform else ', Non-Uniform Knots'
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('B-spline Basis Function Value')
    #plt.legend()
    plt.tight_layout()
    # save figure
    filename = 'Uniform_knots' if uniform else 'Non_uniform_knots'
    filename += '_orthogonal' if orthogonal else '_standard'
    plt.savefig(os.path.join('figures', f'{filename}.png'))
    plt.close()    


def plot_base_distribution():
    # Generate data for the standard normal distribution
    x_normal = np.linspace(-6, 6, 1000)
    y_normal = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x_normal**2)

    # Generate data for the standard logistic distribution
    x_logistic = np.linspace(-6, 6, 1000)
    y_logistic = logistic.pdf(x_logistic)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(x_normal, y_normal, label='Standard Normal')
    plt.plot(x_logistic, y_logistic, label='Standard Logistic')
    plt.title('Standard Normal and Standard Logistic Distributions')
    plt.xlabel('u')
    plt.ylabel('Probability Density Function (PDF)')
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join('figures', 'base_distributions.png'))      




if __name__ == '__main__':
    #create folder for figures if it does not exist    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    plot_single_Bplines(3,uniform=True,orthogonal=True)
    # Bsplines  
    plot_Bsplines(3)

    for comb in [(True,False),(False,False),(False,True),(True,True)]:
        plot_single_Bplines(3,uniform=comb[0],orthogonal=comb[1])
    
    # normflows    
    plot_base_distribution()

