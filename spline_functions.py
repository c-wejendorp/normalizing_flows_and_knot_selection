import torch

def B_spline_torch(x, i, p, knots):
    """
    Implementation of the Cox De Boor algorithm for PyTorch

    Parameters:
    - x: The value at which to evaluate the basis function (PyTorch tensor).
    - i: The index of the basis function.
    - p: The degree of the B-spline.
    - knots: The knot vector (PyTorch tensor).

    Returns:
    - The value of the B-spline basis function at x (PyTorch tensor).
    """
    # Convert inputs to PyTorch tensors
    #x = torch.tensor(x, dtype=torch.float64)
    #knots = torch.tensor(knots, dtype=torch.float64)

    # Base case: If the degree is 0, the basis function is 1 if the parameter is
    # between the i-th and (i+1)-th knot, and 0 otherwise.
    if p == 0:
        return ((knots[i] <= x) & (x < knots[i+1])).to(torch.float64)

    # Recursive case: Apply the Cox-de Boor recursion formula.
    denom1 = knots[i + p] - knots[i]
    term1 = torch.zeros_like(x) if denom1 == 0.0 else ((x - knots[i]) / denom1) * B_spline_torch(x, i, p-1, knots)

    denom2 = knots[i + p + 1] - knots[i+1]
    term2 = torch.zeros_like(x) if denom2 == 0.0 else ((knots[i + p + 1] - x) / denom2) * B_spline_torch(x, i+1, p-1, knots)

    result = term1 + term2

    return result

def B_spline_matrix(x, n, p, knots):
    """
    Generate an mxn matrix where the columns are the discrete evaluatons of the jth B-spline.

    Parameters:
    - x: The vector of values at which to evaluate the basis functions (PyTorch tensor).
    - n: The number of b-splines.
    - p: The degree of the B-spline.
    - knots: The knot vector (PyTorch tensor).

    Returns:
    - An mxn matrix 
    """
   
    m = len(x)
    result_matrix = torch.zeros((m, n), dtype=torch.float32)

    for i in range(n):
        result_matrix[:, i] = B_spline_torch(x, i, p, knots)

    return result_matrix

def gram_schmidt(A):    
    _, m = A.shape
    
    for i in range(m):
        q = A[:, i]  # i-th column of A

        for j in range(i):
            #q = q - torch.dot(A[:, j], A[:, i]) * A[:, j]
            q = q - ( torch.dot(q, A[:, j]) / torch.dot(A[:, j], A[:, j]) ) * A[:, j]


        if torch.all(q.eq(torch.zeros_like(q))):
            pass
            # only goes here on the flow 2 dataset, for some reason.
            #raise torch.linalg.LinAlgError("The column vectors are not linearly independent")

        # normalize q
        #q = q / torch.sqrt(torch.dot(q, q))

        # write the vector back in the matrix
        A[:, i] = q


    return A  # Convert back to NumPy array before returning

