import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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

    def __init__(self, orig_response, layers, fwhm):

        self.layers = layers
        self.orig_response = orig_response
        self.fwhm = fwhm

        if layers!=10:
            self.p2t = self.__calculate_p2t__(6.3, layers, 10)
        else:
            self.p2t = 6.3

        matrix = np.eye(self.layers)
        lower_triangle = np.tril(np.ones((self.layers, self.layers)), k=-1)
        self.p2t_matrix = matrix + (1/self.p2t)*lower_triangle
        
        PSF_3D_matrix = self.__calculate3DPSF_gaussian__(matrix, self.fwhm)
        self.outputMatrix = np.einsum('ij,jkl->ikl', self.p2t_matrix, PSF_3D_matrix)


    def __calculate_p2t__(self, p2t_model, n, n_model):
        
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

    def __calculate3DPSF_gaussian__(self,matrix, fwhm):
        
        fwhm_values = np.linspace(fwhm[0], fwhm[1], matrix.shape[2]-1)  
        sigma_values = fwhm_values / (2 * np.sqrt(2 * np.log(2)))  

        for z in range(matrix.shape[2]-1):
            matrix[:, :, z:z+1] = gaussian_filter1d(matrix[:, :, z:z+1], sigma=sigma_values[z])

        return matrix
