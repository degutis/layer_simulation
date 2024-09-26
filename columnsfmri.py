import numpy as np
#from scipy.stats import norm, chi2, gamma, weibull_min
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class simulation:
    """
    Implementation of a simulation of cortical column patterns and their MR imaging.
    Taken and adjusted from Chaimow et al. (2018) https://github.com/dchaimow/columnsfmri
    """
    def __init__(self,N,L,seed):
        """
        Initializes simulation.
        N   simulation grid points along one dimension
        L   simulation grid length along one dimension [mm]
        """
        self.N = N
        self.L = L
        self.seed = seed # added a seed in the simulation part
        self.dx = L/self.N
        self.dk = 1/self.L
        self.Fs = 1/self.dx
        
        self.x = np.fft.fftshift(
            np.concatenate(
                (np.arange(0,self.N/2),np.arange(-self.N/2,0))) * self.dx)
        self.k = np.fft.fftshift(
            np.concatenate(
                (np.arange(0,self.N/2),np.arange(-self.N/2,0))) * self.dk)
        self.x1, self.x2 = np.meshgrid(self.x, self.x)
        self.k1, self.k2 = np.meshgrid(self.k, self.k)
        
    def gwnoise(self):
        """
        Generates a complex-valued gaussian white noise pattern according to
        simulation grid. To be used for simulating a column pattern using 
        columnPattern.
        """
        np.random.seed(self.seed)
        return np.random.randn(self.N,self.N) + 1j* np.random.randn(self.N,self.N)
    
    def ft2(self,y):
        """
        Numerically simulates 2D fourier transform of patterns defined on the 
        simulation grid.
        """
        return (self.L**2/self.N**2)*np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(y)))
    
    def ift2(self,fy):
        """
        Numerically simulates inverse 2D fourier transform of patterns defined on 
        the simulation grid.
        """
        return (self.N**2 * self.dk**2)*np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(fy)))
    
    def columnPattern(self,rho,deltaRelative,gwnoise):
        """
        Simulates the differential neuronal response of a pattern of cortical 
        columns by filtering of spatial Gaussian white noise gwnoise using a 
        spatial band-pass filter parameterized by main spatial frequency rho and
        relative irregularity delta. Returns the simulated pattern and a map of
        preferred orientaion (if the map is interpreted as representing orientation
        responses).
        """
        fwhmfactor = 2*np.sqrt(2*np.log(2))
        r = np.sqrt(self.k1**2+self.k2**2)
        if deltaRelative==0:
            FORIENTNotNormalized = np.double(abs(r - rho)<self.dk/2)
        else:
            FORIENTNotNormalized = \
            norm.pdf(r,loc= rho,scale=(rho*deltaRelative)/fwhmfactor) + \
            norm.pdf(r,loc=-rho,scale=(rho*deltaRelative)/fwhmfactor)
        C = (np.sqrt(meanpower(FORIENTNotNormalized)))*np.sqrt(np.pi/8)
        FORIENT = FORIENTNotNormalized/C
        noiseF = self.ft2(gwnoise)
        gamma = self.ift2(FORIENT*noiseF)
        neuronal = np.real(gamma)
        preferredOrientation = np.angle(gamma)/2
        return neuronal, preferredOrientation
    
    def bold(self,fwhm,beta,y):
        """
        Simulates spatial BOLD response to neuronal response pattern y using a 
        BOLD PSF with full-width at half-maximum fwhm, response amplitude beta.
        Returns rhe resulting BOLD response pattern, the point-spread function and
        modulation-transfer function.
        """
        
        if fwhm==0:
            by = beta * y
            psf = None
            MTF = np.ones(np.shape(y))
        else:
            fwhmfactor = 2*np.sqrt(2*np.log(2))
            psf = beta * \
            norm.pdf(self.x1,loc=0,scale=fwhm/fwhmfactor) * \
            norm.pdf(self.x2,loc=0,scale=fwhm/fwhmfactor)
            MTF = beta * np.exp(-(self.k1**2+self.k2**2) * 
                                2*(fwhm/fwhmfactor)**2*np.pi**2) 
            by = np.real(self.ift2(MTF*self.ft2(y)))
        return by,psf,MTF
    
    def mri(self,w,y):
        """
        Simulates MRI voxel sampling from pattern y by reducing the k-space
        represenation according to voxel width w, which need to be a devisor 
        of self.L.
        """
        nSamplesHalf = self.L/(2*w)
        if nSamplesHalf % 1 == 0:
            nSamplesHalf = int(nSamplesHalf)
            yk = self.ft2(y)
            centerIdx = int(self.N/2)
            downSampledY = yk[centerIdx-nSamplesHalf:centerIdx+nSamplesHalf,
                              centerIdx-nSamplesHalf:centerIdx+nSamplesHalf]
            my = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(downSampledY))))/w**2
        else:
            my = None
        return my
    
    def upsample(self,y):
        """
        Spatial frequency space zero-fill interpolation (=Fourier interpolation).
        """
        Fy = np.fft.fft2(y)
        nx, ny = Fy.shape # size of original matrix
        nxAdd = self.N - nx # number of zeros to add
        nyAdd = self.N - ny
        # quadrants of the FFT, starting in the upper left
        q1 = Fy[:int(nx/2),:int(ny/2)]
        q2 = Fy[:int(nx/2),int(ny/2):]
        q3 = Fy[int(nx/2):,int(ny/2):]
        q4 = Fy[int(nx/2):,:int(ny/2)]
        zeroPaddRows = np.zeros((nxAdd,int(ny/2)))
        zeroPaddColumns = np.zeros((nxAdd+nx,nyAdd))
        zeroPaddFy = np.hstack(
            (np.vstack((q1,zeroPaddRows,q4)),
             zeroPaddColumns,
             np.vstack((q2,zeroPaddRows,q3))))
        upPattern = np.real(np.fft.ifft2(zeroPaddFy)) * self.N**2/(nx*ny)
        return upPattern
    
    def patternCorrelation(self,orig,mri):
        """
        Calculates pattern correlation between original pattern orig and zero-fill
        interpolated version of pattern mri. 
        """
        mriUp = self.upsample(mri)
        c = np.corrcoef(orig.flatten(),mriUp.flatten())
        r = c[0,1]
        return r
        
    def plotPattern(self,y,cmap='gray',title=None, ax=None):
        """
        Visualizes a simulation pattern.
        y      pattern to be visualized
        cmap   colormap (gray by default)
        title  title
        ax     preexisting axis to be used for plotting
        """
        if not(ax):
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        minx = min(self.x)
        maxx = max(self.x)
        im = ax.imshow(y,cmap,
                       extent=[minx,maxx,minx,maxx],
                       interpolation='none')
        ax.set_title(title)
        ax.set_xlabel('position [mm]')
        fig.colorbar(im,ax=ax)
        plt.savefig('../derivatives/pattern_simulation/'+title.replace(" ","") +'.png')


def noiseModel(V,TR,nT,differentialFlag,*args,**kwargs):
    """
    sigma = noiseModel(V,TR,nT,differentialFlag,...) calculates the
    standard deviation of fMRI noise relative to signal after differential analysis of
    multiple measurements nT (see below) using voxel volume V and TR.
 
    nT is the number of volumes to be averaged:
    it is used to scale the thermal noise factor, assuming thermal noise is
    uncorrelated in time
    AND it is used to scale the physiological noise factor under the
    assuption that physiological noise is a AR(1) process with
    q = exp(-TR/tau), tau = 15 s (Purdon and Weisskoff 1998)
 
    with differential==true flag nT/2 volumes belong to condition A and nT/2
    volumes to condition B
 
    The noise model is based on Triantafyllou et al. 2005.
    It is specified as an additional argument 
     noiseType = {'3T','7T','Thermal','Physiological'}
    or as additional model parameters:
     k,l,T1 corresponding to kappa and lambda in Tiantafyllou et al. 2005 and time constant T1
    """
    TR0 = 5.4
    
    noiseType = kwargs.get('noiseType', None)
    k = kwargs.get('k', None)
    l =  kwargs.get('l', None)
    T1 = kwargs.get('T1', None)
    
    if noiseType == None:
        if l == None or k == None or T1 == None:
            raise ValueError('k,l or T1 not specified!')
    else:
        if l != None or k != None or T1 != None:
            raise ValueError('specify either noiseType or (k,l and T1), not both!')
    if noiseType == '3T':
        k = 6.6567 
        l = 0.0129 
        T1 = 1.607 
    if noiseType =='7T':
        k = 9.9632 
        l = 0.0113 
        T1 = 1.939 
    if noiseType =='Thermal':
        k = 9.9632 
        l = 0 
        T1 = 1.939 
    if noiseType == 'Physiological':
        k = np.Inf 
        l = 0.0113 
        T1 = 1.939 
    
    if not(differentialFlag) and nT != 1:
        raise ValueError('for multiple measurements only differential implemented!')
    elif nT == 1:
        F = np.sqrt(np.tanh(TR/(2*T1))/np.tanh(TR0/(2*T1)))
        k = k * F
        sigma = np.sqrt(1+l**2*k**2*V**2)/(k*V)
        return sigma
    
    s = 0
    assert nT%2==0 
    for t1 in range(1,int(nT/2)+1):
        for t2 in range(1,int(nT/2)+1):
            s = s + np.exp((-TR*abs(t1-t2))/15)
    
    F = np.sqrt(np.tanh(TR/(2*T1))/np.tanh(TR0/(2*T1)))
    k = k * F
    sigma = np.sqrt((4/(k**2*V**2*nT)) + ((2*l**2)/(nT/2)**2)*s)
    return sigma