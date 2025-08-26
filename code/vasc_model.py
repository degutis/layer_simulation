import numpy as np
from scipy.stats import norm

class vascModel:
    """
    A class to generate a response affected by vein draining as proposed by Markuerkiaga et al. (2021) based on the vascular model
    proposed in Markuerkiaga et al. (2016).
    ...

    Attributes
    ----------
    layers : int
        the number of layers/bins that are estimated
    p2t : float
        peak-to-tail ratio. The value used to calculate the off-diagonal elements in the lower triangle of the matrix X  (PSF matrix)
    matrix : numpy.ndarray
        the resulting matrix after the operation
    original_response : float
        the original response value used to calculate the output matrix
    outputMatrix : numpy.ndarray
        the final output matrix after multiplying the original response value

    Methods
    -------
    __calculate_p2t__(base, layers, base_layers):
        Calculates the value of p2t based on the base value, number of layers, and base number of layers.
    """

    def __init__(self, orig_response, layers, fwhm, propChange = 0):

        self.layers = layers
        self.orig_response = orig_response
        self.fwhm = fwhm

        if layers!=10:
            self.p2t = __calculate_p2t__(6.3, layers, 10)
        else:
            self.p2t = 6.3

        self.p2t += propChange*self.p2t

        matrix = np.eye(self.layers)
        lower_triangle = np.tril(np.ones((self.layers, self.layers)), k=-1)
        self.p2t_matrix = matrix + (1/self.p2t)*lower_triangle
        self.PSF_3D_matrix = self.__calculate3DPSF_gaussian__(matrix, self.fwhm)
        self.tranformationMatrix = np.dot(self.p2t_matrix,self.PSF_3D_matrix)
        
        self.outputMatrix = np.einsum('ij,jkl->ikl', self.p2t_matrix, orig_response)

    def __calculate_3DPSF_matrix__(self,matrix, num):
        
        x = matrix.shape[0]

        matrix += np.diag([num] * (x - 1), k=1)  
        matrix += np.diag([num] * (x - 1), k=-1) 

        matrix += np.diag([num / 2] * (x - 2), k=2)  
        matrix += np.diag([num / 2] * (x - 2), k=-2) 


        top_triangle = np.triu(np.ones((x,x)), k=1)
        gradient_values = np.linspace(0, num, np.sum(top_triangle == 1))
        top_triangle[top_triangle == 1] = gradient_values
        np.fill_diagonal(top_triangle, 1)
        combined = np.dot(matrix,np.round(top_triangle,3))
        np.fill_diagonal(combined, 1)

        return np.dot(matrix,top_triangle)

    def __calculate3DPSF_gaussian__(self,matrix, fwhm, diagonalOne = False):
        
        matrix_size = matrix.shape[0]
        fwhm_values = np.linspace(fwhm[0], fwhm[1], matrix_size)  
        sigma_values = fwhm_values / (2 * np.sqrt(2 * np.log(2)))  
        transformation_matrix = np.zeros((matrix_size, matrix_size))

        for i, sigma in enumerate(sigma_values):
            x = np.arange(matrix_size)
            gaussian = norm.pdf(x, loc=i, scale=sigma)  
            transformation_matrix[i, :] = gaussian/gaussian.sum()
        
        if diagonalOne:
            for col in range(transformation_matrix.shape[1]):
                max_val = np.max(transformation_matrix[:, col])
                if max_val > 0: 
                    transformation_matrix[:, col] /= max_val

        return transformation_matrix

def __calculate_p2t__(p2t_model, n, n_model):
        
    """
    Calculate the value of p2t and round to the nearest two decimals:
    
    Parameters:
    p2t_model : float : Original p2t value
    n : float : New layer value
    n_model : float : Original layer value
    
    Returns:
    float : The calculated value of p2t
    """
    return np.round((n / n_model) * p2t_model + (n_model - n) / (2 * n_model),2)


def deconvolve(orig_response, propChange=0):

    layers = orig_response.shape[2]

    if layers!=10:
        p2t = __calculate_p2t__(6.3, layers, 10)
    else:
        p2t = 6.3


    p2t += propChange*p2t

    matrix = np.eye(layers)
    lower_triangle = np.tril(np.ones((layers, layers)), k=-1)
    p2t_matrix = matrix + (1/p2t)*lower_triangle

    inv_transformation_matrix = np.linalg.pinv(p2t_matrix)
    deconvolved_response = np.einsum('ij,jkl->ikl', inv_transformation_matrix, np.transpose(orig_response, (2,1,0))).transpose(2,1,0)    
    
    return deconvolved_response
