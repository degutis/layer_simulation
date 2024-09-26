import numpy as np

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

    def __init__(self, orig_response, layers=10):

        self.layers = layers

        if layers!=10:
            self.p2t = self.__calculate_p2t__(6.3, self.layers, 10)
        else:
            self.p2t = 6.3

        self.matrix = np.eye(self.layers)
        lower_triangle = np.tril(np.ones((self.layers, self.layers)), k=-1)
        self.matrix = self.matrix + (1/self.p2)*lower_triangle

        self.outputMatrix = self.matrix * self.original_response

    def __calculate_p2t__(p2t_model, n, n_model):
        
        """
        Calculate the value of p2t and round to the nearest two decimals:
    
        Parameters:
        p2t_model : float : The value of p2t_model
        n : float : The value of n
        n_model : float : The value of n_model
    
        Returns:
        float : The calculated value of p2t
        """
        return np.round((n / n_model) * p2t_model + (n_model - n) / (2 * n_model),2)


