import numpy as np
#from scipy.stats import norm, chi2, gamma, weibull_min
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.fft import fftn, ifftn

class simulation:
    """
    Implementation of a simulation of cortical column patterns and their MR imaging.
    Taken and adjusted from Chaimow et al. (2018) https://github.com/dchaimow/columnsfmri
    """
    def __init__(self,N,L,N_depth, L_depth, seed):
        """
        Initializes simulation.
        N   simulation grid points along one dimension
        L   simulation grid length along one dimension [mm]
        """
        self.N = N
        self.L = L
        self.N_depth = N_depth
        self.L_depth = L_depth
        self.seed = seed # added a seed in the simulation part
        self.dx = L/self.N
        self.dk = 1/self.L
        self.Fs = 1/self.dx

        self.dz = L_depth/self.N_depth

        self.x = np.fft.fftshift(
            np.concatenate(
                (np.arange(0,self.N/2),np.arange(-self.N/2,0))) * self.dx)
        self.k = np.fft.fftshift(
            np.concatenate(
                (np.arange(0,self.N/2),np.arange(-self.N/2,0))) * self.dk)

        self.z = np.fft.fftshift(np.arange(0, self.N_depth) * self.dz)
        self.x1, self.x2, self.x3 = np.meshgrid(self.x, self.x, self.z)
    
        self.k1, self.k2 = np.meshgrid(self.k, self.k)
        
    def gwnoise(self):
        """
        Generates a complex-valued gaussian white noise pattern according to
        simulation grid. To be used for simulating a column pattern using 
        columnPattern.
        """
        rng = np.random.RandomState(self.seed)
        return rng.randn(self.N,self.N) + 1j* rng.randn(self.N,self.N)
    
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
    
    def ft3(self, y):
        """
        Numerically simulates 3D Fourier transform of patterns defined on the 
        simulation grid for the x, y dimensions (L) and z dimension (L_depth).
        """
        return (self.L**2 * self.L_depth / (self.N**2 * self.N_depth)) * np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(y), s=(self.N, self.N, self.N_depth))
        )
   
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
    
    def mri(self, w, y):
        """
        Simulates MRI voxel sampling from a 3D pattern y by reducing the k-space
        representation according to voxel width w, which needs to be a divisor 
        of L for x and y dimensions, and L_depth for the z dimension.
        """
        nSamplesHalfXY = self.L / (2 * w)      # Down-sample in x and y dimensions
        nSamplesHalfZ = self.L_depth / (2 * w) # Down-sample in z dimension based on L_depth

        if nSamplesHalfXY % 1 == 0 and nSamplesHalfZ % 1 == 0:
            nSamplesHalfXY = int(nSamplesHalfXY)
            nSamplesHalfZ = int(nSamplesHalfZ)

            yk = self.ft3(y)  # Use the 3D Fourier transform with L_depth and N_depth

            centerIdxXY = int(self.N / 2)       # Center index for x and y dimensions
            centerIdxZ = int(self.N_depth / 2)  # Center index for z dimension (using N_depth)

            # Downsample in all three dimensions
            downSampledY = yk[
                centerIdxXY - nSamplesHalfXY : centerIdxXY + nSamplesHalfXY,
                centerIdxXY - nSamplesHalfXY : centerIdxXY + nSamplesHalfXY,
                centerIdxZ - nSamplesHalfZ : centerIdxZ + nSamplesHalfZ
            ]

            # Perform inverse FFT and normalize
            my = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(downSampledY)))) / (w**3)
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
        #plt.savefig('../derivatives/pattern_simulation/'+title.replace(" ","") +'.png')


    def noiseModel(self, V,TR,nT,differentialFlag,*args,**kwargs):
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
        
        physNoiseSpatialWidth = 0.01

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
        
        else:
            s = 0
            assert nT%2==0 
            for t1 in range(1,int(nT/2)+1):
                for t2 in range(1,int(nT/2)+1):
                    s = s + np.exp((-TR*abs(t1-t2))/15)
            
            F = np.sqrt(np.tanh(TR/(2*T1))/np.tanh(TR0/(2*T1)))
            k = k * F
            sigma = np.sqrt((4/(k**2*V**2*nT)) + ((2*l**2)/(nT/2)**2)*s)

        physNoiseFilter = norm.pdf(np.sqrt(self.x1**2 + self.x2**2 + self.x3**2), 0, 
                        physNoiseSpatialWidth / (2 * np.sqrt(2 * np.log(2))))
        FphysNoiseFilterNotNormalized = fftn(physNoiseFilter) * self.dx
        FphysNoiseFilter = FphysNoiseFilterNotNormalized / np.sqrt(meanpower(FphysNoiseFilterNotNormalized))   
        Fnoise = fftn(np.random.randn(*self.x1.shape)) * self.dx
        physNoise = l * ifftn(Fnoise * FphysNoiseFilter) * self.dk

        return sigma, np.real(physNoise)

def meanpower(s):
    """
    calculates the mean power of a signal or pattern
    """
    return np.mean(np.abs(s**2))
