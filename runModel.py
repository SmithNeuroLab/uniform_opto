#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:11:44 2022

@author: haleigh
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg



def runModel(networkSize=(30,30),nevents=40,index=0,add_noise=True,structured_input=None,structured_amp=0.01,strength_deformation=0.2):
    #Linear model with perturbed, homogenous difference of Gaussians connectivity (aka homogenous MH). 
    #Can accept structured input to simulate optic/optogenetic spatially specific drive.
    # Code adapted from Sigrid Tragenap 
    # Haleigh Mulholland, 06/30/2022
    
    # INPUT
    #   networkSize=N,M 
    #   nevents=Number of patterns to generate
    #   index=Seed for random number generator 
    #   add_noise=If true, will add noise to the initial condition. Without noise, model output is essentially mean response
    #   structured_input=A NxM array of patterned activity to be added on top of input, scaled so min=0, max=1. Used to simulate optic/optogenetic drive. If None, input is just bandpass noise.     
    #   structured_amp=Multiplier for strucutred input
    #   strength_deformation= heterogeneity coefficient. Range 0-1, with 0 being a homogenous DoG. 
    
    # OUTPUT
    #   activity, inputs
    
    
    N=networkSize[0]
    M=networkSize[1]
    R=0.99  #The closer the factor is to 1, the smaller the dimensionality the patterns it produces
    

    # #use deformed MH
    w_rec_unscaled = homogenMH_deformed_wrap(N,M,sigmax=1.8,a1=1,inh_factor=2,
                      sigma1_bp=2, sigma2_bp=6, strength_h=strength_deformation,outputshape='2d',seed=index)
    w_rec = scale_eigenvalue(w_rec_unscaled, R)
    
    
    if add_noise:
        #create input
        rng_start_activity = np.random.RandomState(index)
        input_noise_level=0.3
        
        additive_noise=rng_start_activity.normal(0,1,(N*M*nevents))
        additive_noise=additive_noise.reshape(nevents,N*M)
        additive_noise=additive_noise*input_noise_level
    else:
        additive_noise=np.zeros((nevents,N*M))

    #Add structured input
    if structured_input is not None:
        print('Adding structured input')
        str_inp=structured_input*structured_amp
        str_inp=str_inp.reshape((N*M))
        constant_input = np.ones((N*M))
        #constant_input=str_inp
        constant_input=constant_input+str_inp
    else:
        constant_input = np.ones((N*M))
    
    iinputs=constant_input+additive_noise
    iinputs=iinputs.T
    
    res_output= np.dot(np.linalg.inv(np.identity(w_rec.shape[0])-w_rec),iinputs)

    #reshape into patterns
    activity = np.empty((nevents, N, M))
    inputs = np.empty((nevents, N,M))
    for iact in range(nevents):
        activity[iact,:,:]=res_output[:,iact].reshape(N,M)
        inputs[iact,:,:]=iinputs[:,iact].reshape(N,M)

    return activity, inputs

#%%
def noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,orientation_sd,\
a1,inh_factor,pbc=True,index=4876,full_output=True,rotate_by=None,conv_params=None):
    coord_x,coord_y= np.meshgrid(np.arange(N),np.arange(M)) #N:x dimension; M:y dim.

    np.random.seed(index)


    ## periodic boundary conditions
    if pbc:
        deltax = coord_x[:,:,None,None]-coord_x[None,None,:,:]
        deltay = coord_y[:,:,None,None]-coord_y[None,None,:,:]
        absdeltax = np.abs(deltax)
        absdeltay = np.abs(deltay)
        idxx = np.argmin([absdeltax, N-absdeltax],axis=0)
        idxy = np.argmin([absdeltay, M-absdeltay],axis=0)

        deltax = deltax*(1-idxx) + np.sign(deltax)*(absdeltax-N)*idxx
        deltay = deltay*(1-idxy) + np.sign(deltay)*(absdeltay-M)*idxy
    else:
        deltax = coord_x[:,:,None,None]-coord_x[None,None,:,:]
        deltay = coord_y[:,:,None,None]-coord_y[None,None,:,:]




    if mode=='short_range':
        if 'None' in noise_type:
            '''homogeneous mhs'''
            sigmax=conv_params['s1_x']
            sigmay=conv_params['s1_x']
            delta = (deltax)**2/sigmax**2 + (deltay)**2/sigmay**2

            mh1=gauss(delta,1.)/2./np.pi
            mh2=gauss(delta,inh_factor)/2./np.pi
            anisotropic_mh = ( mh1 - a1*mh2 )
        elif noise_type=='postsyn':
            ## generate spatial scale in x direction
            if conv_params['do_convolution_x']:
                sigmax_noise = convolve_with_MH(np.random.randn(M,N),N,M,sigma1=conv_params['s1_x'],sigma2=conv_params['s2_x'])[:,:,None,None]
            else:
                sigmax_noise = np.random.randn(M,N)[:,:,None,None]
            sigmax_noise = (sigmax + sigmax_noise/np.std(sigmax_noise)*sigmax_sd)
            sigmax_noise[sigmax_noise<0]=0.0

            ## generate eccentricity array
            if conv_params['do_convolution_ecc']:
                ecc_noise = convolve_with_MH(np.random.randn(M,N),N,M,sigma1=conv_params['s1_ecc'],sigma2=conv_params['s2_ecc'])[:,:,None,None]
            else:
                ecc_noise = np.random.randn(M,N)[:,:,None,None]
            ecc_noise = ecc + ecc_noise/np.std(ecc_noise)*ecc_sd

            #rewrite using clip
            ecc_noise[ecc_noise>0.95] = 0.95
            ecc_noise[ecc_noise<0.0] = 0.0

            ## calculate spatial scale in y-direction
            sigmay_noise = sigmax_noise*np.sqrt(1 - ecc_noise**2)

            ## generate array of orientations
            z_noise = np.random.randn(M,N) + 1j*np.random.randn(M,N)
            if conv_params['do_convolution_ori']:
                z_noise = convolve_with_MH(z_noise,N,M,sigma1=conv_params['s1_ori'],sigma2=conv_params['s2_ori'],return_real=False,padd=conv_params['padd'])
            elif conv_params['const_ori']:
                z_noise = np.zeros((M,N),dtype='complex')
            orientation_noise = np.angle(z_noise)*0.5
            orientation_noise = orientation + orientation_noise

            orientation_noise = orientation_noise + np.pi*(orientation_noise<(np.pi/2))
            orientation_noise = orientation_noise - np.pi*(orientation_noise>(np.pi/2))

            cos_noise = np.cos(orientation_noise)[:,:,None,None]
            sin_noise = np.sin(orientation_noise)[:,:,None,None]

            if not full_output:
                return ecc_noise[:,:,0,0],orientation_noise,sigmax_noise[:,:,0,0],sigmay_noise[:,:,0,0]

            delta = (deltax*cos_noise - deltay*sin_noise)**2/sigmax_noise**2 + (deltay*cos_noise + sin_noise*deltax)**2/sigmay_noise**2
            mh1 = gauss(delta,1.)/sigmay_noise/sigmax_noise/2./np.pi
            mh2 = gauss(delta,inh_factor)/sigmay_noise/sigmax_noise/2./np.pi
            anisotropic_mh = ( mh1 - a1*mh2 )



        w_rec=np.real(anisotropic_mh)

        if conv_params['do_Binomial_sampling']:
            rng=np.random.RandomState(seed=index)
            m1_sampled=rng.binomial(conv_params['K_E'], mh1)*conv_params['w_E']
            m2_sampled=rng.binomial(conv_params['K_I'], a1*mh2)*conv_params['w_I']
            w_rec=m1_sampled+m2_sampled

        return w_rec
    
    
def homogenMH_deformed_wrap(N,M,sigmax,a1=1,inh_factor=2,
                  pbc=True,index=4876,full_output=True,  #MH params
                  sigma1_bp=2, sigma2_bp=6, strength_h=0.4, seed=None,  #random field params
                  outputshape='2d'):

    MH_connectivity = noisy_mh(N,M,
                 mode='short_range',
                 noise_type="None",
                 sigmax=sigmax,
                 sigmax_sd=0,
                 ecc=0,
                 ecc_sd=0,orientation=0,
                 orientation_sd=0,
                 a1=a1,
                 inh_factor=inh_factor,
                 pbc=pbc,
                 index=index,
                 full_output=full_output,
                 rotate_by=None,
                 conv_params={'s1_x': sigmax,
                              'do_Binomial_sampling': False})
    
    #Deformed
    #add bandpass noise
    randomGaussFields = get_random_field(N,M,N_fields=N*M,
                                         sig1=sigma1_bp, sig2=sigma2_bp,
                                         seed=seed)
    
    
    # Perturbation matrix [1+hG]_+  , where G is NxN Gaussian noise bandpass filtered
    # #Use the same random gaussian field for each neuron
    perturbMatrix=1+randomGaussFields[0,:,:]*strength_h
    perturbMatrix[perturbMatrix<0]=0
    
    connectivity=np.zeros((N,M,N,M))
    for ix in range(N):
        for iy in range(M):
            connectivity[ix,iy,:,:]=MH_connectivity[ix,iy,:,:] * perturbMatrix


    if outputshape=='2d':
        w_rec = connectivity.reshape(N*M, N*M)
    elif outputshape=='4d':
        w_rec = connectivity.copy()
    return w_rec

def gauss(delta,inh_factor):
    return 1./inh_factor**2*np.exp(-delta/2./inh_factor**2)

def get_random_field(N,M,N_fields=1, sig1=2, sig2=4, seed=None):
    #constant input in time!
    #N, M : grid size
    #N_fields: number of fields
    #sig1, sig2: kernel size

    #generate random input
    rng_fields = np.random.default_rng(seed)
    #sample from normal distribution, other distributions possible
    input_rnd = rng_fields.normal(size=[N_fields, N, M])

    if np.allclose(0,sig1):
        bp_field = np.copy(input_rnd)
    else:
        #apply convolution
        #use convolution with MH to get spatial scale in noisy input
        #define convolutio kernels
        x,y = np.meshgrid(np.linspace(-N//2+1,N//2,N),np.linspace(-M//2+1,M//2,M))
        kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
        kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
        input_smo = np.real(np.fft.ifft2(np.fft.fft2(kern1-kern2)[None,:,:]*np.fft.fft2(input_rnd,axes=(1,2)), axes=(1,2)))
        bp_field = input_smo.copy()

    return bp_field

def convolve_with_MH(input_rnd,N,M,sigma1=2,sigma2=6,return_real=True,padd=0):
    ''' use convolution with MH to get spatial scale in noisy input'''
    h,w = input_rnd.shape
    x,y = np.meshgrid(np.linspace(-N//2+1,N//2,N+2*padd),np.linspace(-M//2+1,M//2,M+2*padd))
    sig1 = sigma1#2
    sig2 = sigma2#3*sig1
    kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
    kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
    diff_gauss = kern1-kern2
    to_smooth = np.pad(input_rnd,padd,'constant')
    input_smo = np.fft.ifft2(np.fft.fft2(diff_gauss)*np.fft.fft2(to_smooth,axes=(0,1)), axes=(0,1))
    input_smo = np.fft.fftshift(input_smo)
    hn,wn = input_smo.shape
    if padd>0:
        input_smo = input_smo[hn//2-h//2:hn//2+h//2,wn//2-w//2:wn//2+w//2]
    if return_real:
        return np.real(input_smo)
    else:
        return input_smo


#scale by max eigenvalue
def scale_eigenvalue(w_rec,R):
    ## normalize M such that real part of maximal eigenvalue is given by network_params['nonlin_fac']
    all_eigenvals =linalg.eigvals(w_rec)
    max_eigenval = np.nanmax(np.real(all_eigenvals))
    w_rec = R*w_rec/np.real(max_eigenval)
    return w_rec

def calcSpatialScale(sigmax=None,inh_factor=None,**kwargs):
    '''Calculates the typical spatial scale in microns per pixel.
    
    This is eqn 20 in manuscript (Smith et al 2018, Nat Neuro)
    Here: sigmax is sigma_1 in the paper, inh_factor is kappa in the paper
    
    This is spatial scale is typically thought to be equivalent to 1000 um
    '''
    return(1000./np.sqrt((4*np.pi**2*sigmax**2 * (inh_factor**2-1))/(4*np.log(inh_factor))))

def generateRandomBandpassPatterns(low,high,imgSize=(60,60),sigma=None,npatterns=1,seed=81622):

    np.random.seed(seed)
    patterns=[]
    for ipattern in range(npatterns):
        img_c1 = np.random.normal(0, 1, size=imgSize)
        img_c2 = np.fft.fft2(img_c1)
        img_c3 = np.fft.fftshift(img_c2) 
        if sigma is not None:
            img_c3=img_c3*gaussFilterBP(low,high,sigma,img_c1.shape)
        else:
            img_c3=img_c3*idealFilterBP(low,high,img_c1.shape)
        img_c4 = np.fft.ifftshift(img_c3)
        img_c5 = np.fft.ifft2(img_c4)
        patterns.append(img_c5.real)
        ## Visualize spectrum
        # plt.figure(figsize=(8,4))
        # plt.subplot(151), plt.imshow(img_c1, "gray"), plt.title("Original Image")
        # plt.subplot(152), plt.imshow(np.log(1+np.abs(img_c2)), "gray"), plt.title("Spectrum")
        # plt.subplot(153), plt.imshow(np.log(1+np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
        # plt.subplot(154), plt.imshow(np.log(1+np.abs(img_c4)), "gray"), plt.title("Decentralized")
        # plt.subplot(155), plt.imshow(img_c5.real, "gray"), plt.title("Processed Image")
        # plt.tight_layout()
        # plt.show()
        
    return patterns

def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterBP(low_bp,high_bp,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if (distance((y,x),center) < high_bp) and (distance((y,x),center) > low_bp):
                base[y,x] = 1
    return base

def gaussFilterBP(low_bp,high_bp,sigma,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if (distance((y,x),center) < high_bp) and (distance((y,x),center) > low_bp):
                base[y,x] = 1
                
    base2=smooth_map.lowhigh_normalize(base,sig_high=sigma+1,sig_low=sigma)
    base2[base2<=0]=0
    #scale
    min1=np.min(base2)
    max1=np.max(base2)
    base2=(base2-min1)/(max1-min1)
    base2[np.array(base,dtype='bool')]=1
    # plt.figure()
    # plt.imshow(base2)
    
    return base2


