import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, orig_response, layers):

        self.layers = layers
        self.orig_response = orig_response

        if layers!=10:
            self.p2t = self.__calculate_p2t__(6.3, layers, 10)
        else:
            self.p2t = 6.3

        self.matrix = np.eye(self.layers)
        lower_triangle = np.tril(np.ones((self.layers, self.layers)), k=-1)
        self.matrix = self.matrix + (1/self.p2t)*lower_triangle
        self.outputMatrix = np.einsum('ij,jkl->ikl', self.matrix, self.orig_response)



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

    def plotLaminarResp(self, y, FigTitle):

        sizeSquare = int(np.sqrt(self.outputMatrix.shape[1]))
        patternA = np.mean(self.outputMatrix[:,:,np.ravel(y)==0],axis=2).reshape(self.layers,sizeSquare,sizeSquare)
        patternB = np.mean(self.outputMatrix[:,:,np.ravel(y)==1],axis=2).reshape(self.layers,sizeSquare,sizeSquare)

        fig, axs = plt.subplots(2, self.layers, figsize=(15, 10), sharex='col', sharey='row')
        fig.text(0.07, 0.7, 'Pattern 1', va='center', rotation='vertical', fontsize=14)
        fig.text(0.07, 0.3, 'Pattern 2', va='center', rotation='vertical', fontsize=14)
        
        cbar_ax = fig.add_axes([1.05, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='gray'), cax=cbar_ax)

        for i in range(self.layers):
            axs[0, i].imshow(patternA[i], cmap='gray')
            axs[0, i].set_title(f'Layer {i+1}')

            axs[1, i].imshow(patternB[i], cmap='gray')

        fig.savefig(f'../derivatives/laminarPattern/LaminarResponse_{FigTitle}.png',format="png")
        plt.close(fig)

