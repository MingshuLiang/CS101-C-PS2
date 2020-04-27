#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
'''
Example solution script for CS101-C
-----------------------------------

Install the dependencies by running
`python3 -m pip install --user numpy==1.18 imageio==2.8`. You can also use
virtual environments or a tool like [Poetry](https://python-poetry.org/) to
manage your dependencies.

Run `python3 solve.py` to generate answers for everything.

Run `python3 -c "import solve; solve.answer_q1()"` to generate your answer to
question 1. Replace "q1" with "q2" etc. to test the rest of your solutions.
'''
import imageio
import numpy as np
import math
import string
from pathlib import Path
from typing import Tuple, Union
from PIL import Image
from ps_lib import read_image, write_image, pad, resize, _resize_channel

#------------------------------------------------------------------------------
# Define an answer generator for each question.

def answer_q1() -> None:
    print('I\'m answering question 1 now!')
    cdir = str(Path().absolute())
    readpath = cdir + '/apple.png'
    I_o = read_image(readpath)
    
    kernel_G = np.load('gaussian-kernel.npy')
    kernel_S_h = np.load('sobel-kernel-horizontal.npy')
    kernel_S_v = np.load('sobel-kernel-vertical.npy')
    
    I_G = MyConv(I_o,kernel_G)
    write_image(cdir+'/q1-gaussian-kernel.png', I_G)
    
    I_S_h = MyConv(I_o,kernel_S_h)
    write_image(cdir+'/q1-sobel-kernel-horizontal.png', I_S_h)

    I_S_v = MyConv(I_o,kernel_S_v)
    write_image(cdir+'/q1-sobel-kernel-vertical.png', I_S_v)

    
    
    
    
def answer_q2() -> None:
    print('I\'m answering question 2 now!')  
    cdir = str(Path().absolute())
    readpath = cdir + '/mask.png'
    I_o = read_image(readpath) 
    kernel = np.load('gaussian-kernel.npy')
    
    pyramid = Pyramid(I_o, kernel, 'G')
    np.save('q2-mask.npy',pyramid)
    
    #visualization
    for i in range(len(pyramid)):
        pyramid[i] *= 2**i
        write_image(cdir+'/q2-mask-'+str(i+1)+'.png',pyramid[i])
        
        
        
        
        
def answer_q3() -> None:
    print('I\'m answering question 3 now!')
    cdir = str(Path().absolute())
    pyramid_1 = np.load('q2-cat.npy', allow_pickle = True)
    pyramid_2 = np.load('q2-bird.npy', allow_pickle = True)
    mask = np.load('q2-mymask.npy', allow_pickle = True)
    (height, width, channel) = np.shape(pyramid_1[0])
    I_blended = np.zeros((height, width, channel)) 
    for i in range(len(pyramid_1)):
        print(np.shape(mask[i]))
        print(np.shape(pyramid_1[i]))
        print(np.shape(pyramid_2[i]))
        I_blended += (mask[i]*pyramid_1[i] + (1-mask[i])*pyramid_2[i])
        if i < len(pyramid_1)-1:
            (height, width, _) = np.shape(pyramid_1[i+1])
            I_blended  = resize(I_blended,(int(height), int(width)))
        
    write_image(cdir+'/q3-new-images.png', I_blended)
    
    
    
    
    
def answer_q4() -> None:
    print('I\'m answering question 4 now!')
    
    exptime = ['1', '4', '16', '32']
    
    cdir = str(Path().absolute())
    for i in range(4):
        filename = cdir+'/captured-image6-'+exptime[i]+'ms.npz'
        imagedata = np.load(filename)
        I = imagedata['data']
        I = np.flip(I)
        I -= np.min(I)
        I /= (np.max(I))
        write_image(cdir+'/q4-shot-'+str(i+1)+'.png', I)
        
        
        
        
        
def answer_q5() -> None:
    print('I\'m answering question 5 now!')
    cdir = str(Path().absolute())
    I = [[],[],[]]*4
    for i in range(4):
        filename = cdir+'/q4-shot-'+str(i+1)+'.png'
        I[i] = read_image(filename)
    
    (height, width, _)=np.shape(I[0])
    
    gain = np.ones((4,))
    gain_iter = 1
    for i in range(1,4):
        gain_iter *= GetGain(I[i],I[i-1])
        gain[i] = gain_iter
    
    I_HDR = np.zeros(np.shape(I[0]))   
    
    for h in range(height):
        for w in range(width):           
            valid_pix = []
            over_exp = False
            under_exp = False
            min_frame = 3
            for i in range(4):
                if max(I[i][h,w,:])<=0.05:
                    under_exp = True
                elif min(I[i][h,w,:])>=0.95:
                    over_exp = True
                    min_frame = min(min_frame,i)
                else:
                    valid_pix.append(I[i][h,w,:]/gain[i])
            #print(valid_pix)
            if len(valid_pix) == 0:
                if over_exp: 
                    I_HDR[h,w,:] = I[min_frame][h,w,:]/gain[min_frame]
                else: 
                    I_HDR[h,w,:] = 0
            else:
                I_HDR[h,w,:] = np.mean(valid_pix,axis = 0)
            #print(I_HDR[h,w,:])
    np.save('q5-captured-composite.npy',I_HDR)
    
    # Render compisite images
    for i in range(5):
        savepath = cdir+'/q5-captured-rendering-'+str(i+1)+'.png'
        scale = 1.5**i
        write_image(savepath,I_HDR*scale)
        
        
        
        
        
def answer_q6() -> None:
    print('I\'m answering question 6 now!')
    
    cdir = str(Path().absolute())
    readpath = cdir + '/noisy-image.png'
    I_o = read_image(readpath)
    
    ksize = 5
    
    sigma_d = 1
    sigma_r = 0.3
    
    I_bila = BilaFilter(I_o, ksize, sigma_d, sigma_r)
    
    write_image(cdir + '/q6-filtered-image.png', I_bila)

    
    
    
    
def answer_q7() -> None:
    print('I\'m answering question 7 now!')
    cdir = str(Path().absolute())
    I_HDR = np.load('q5-captured-composite.npy')
    
    (height, width, channel) = np.shape(I_HDR)
    
    Chrome = np.zeros(np.shape(I_HDR))
    I_mapped = np.zeros(np.shape(I_HDR))
    
    Y = 0.3*I_HDR[:,:,0] +0.6*I_HDR[:,:,1] + 0.1*I_HDR[:,:,2] + 1e-8
    Chrome[:,:,0] = I_HDR[:,:,0]/Y
    Chrome[:,:,1] = I_HDR[:,:,1]/Y
    Chrome[:,:,2] = I_HDR[:,:,2]/Y
    
    Y_log = np.log10(Y)
    
    ksize = 9
    sigma_d = int(width/50)
    sigma_r = 0.4
    I_base = BilaFilter(Y_log, ksize, sigma_d, sigma_r)
    I_detail = Y_log-I_base
    
    r_target_log = 2
    alpha_base =  r_target_log/(np.max(Y_log)-np.min(Y_log[Y_log> -8]))
    beta = -np.max(Y_log)*alpha_base
    
    alpha_detail = 3
    
    Y_mapped_log = alpha_base*I_base + alpha_detail*I_detail +beta
    Y_mapped_lin = 10**Y_mapped_log
    
    I_mapped[:,:,0] = Chrome[:,:,0]*Y_mapped_lin
    I_mapped[:,:,1] = Chrome[:,:,1]*Y_mapped_lin
    I_mapped[:,:,2] = Chrome[:,:,2]*Y_mapped_lin
    
    write_image(cdir+'/q7-captured.png', I_mapped)
#------------------------------------------------------------------------------
# Define support functions.

# Convolution function for Q1
def MyConv(I_o, kernel):

    kernel_h,kernel_w = np.shape(kernel)
    
    (I_h,I_w,I_ch) = np.shape(I_o) # h-height, w-width, ch-channel
    
    # zero padding
    padding_h = int((kernel_h-1)/2) 
    padding_w = int((kernel_w-1)/2)   
    I = pad(I_o, padding_h, padding_w)
    
    I_f = np.zeros((I_h,I_w,I_ch)) # f-filtered

    
    for i in range(I_h):
        for j in range(I_w):
            for ch in range(I_ch):
                I_cropped = I[i:i+kernel_h,j:j+kernel_w,ch]
                I_f[i][j][ch] = np.sum(kernel*I_cropped)
    
    return I_f





# Pyramid function for Q2 
def Pyramid(I_o, kernel, mode):
    (height, width, channel) = np.shape(I_o)
    (ksize, _) = np.shape(kernel)
    pyramid_G = []
    pyramid_G.append(I_o)
    pyramid_L = []
    I = I_o
    i = 0
    while(min(height, width)>ksize):
        I_filtered = MyConv(I, kernel)
        height/=2
        width/=2
        I_ds = resize(I_filtered,(int(height),int(width)))        
        pyramid_G.append(I_ds)
        I = I_ds
        i += 1
        if mode == 'G':
            continue
        else:
            I_us = resize(I_ds,(int(height*2),int(width*2)))
            pyramid_L.append(pyramid_G[i-1]-I_us)
            if min(height, width)<=ksize:
                pyramid_L.append(I_ds)
    if mode == 'G':
        return np.asarray(pyramid_G[::-1])
    else:
        return np.asarray(pyramid_L[::-1])
    
    
    
    
    
# Gain function for Q5
def GetGain(I2, I1) -> float:
    
    gain_rgb = np.zeros((3,))
    for ch in range(3):        
        I1_mono = I1[:,:,ch]
        I2_mono = I2[:,:,ch]
        
        Valid1 = GetValid(I1_mono)
        Valid2 = GetValid(I2_mono)
    
        V1set = set([tuple(x) for x in Valid1])
        V2set = set([tuple(x) for x in Valid2])
    
        Valid12 = np.array([x for x in V1set & V2set])
        #print(len(Valid12))
        
        gain_rgb[ch] = np.mean(I2_mono[Valid12[:,0],Valid12[:,1]]/I1_mono[Valid12[:,0],Valid12[:,1]])
    
    gain = np.mean(gain_rgb)
    print(gain)
    
    return gain





# Getting valid pixel function for Q5
def GetValid(I):
    
    V = np.argwhere((I>0.05) & (I<0.95));
    
    return V





# Gaussian Function for Q6
def MyGaussian(d,sigma):
    
    x = np.exp(-0.5*(d/sigma)**2)
    
    return x





# Bilateral filtering function for Q6 and Q7    
def BilaFilter(I_o, ksize, sigma_d, sigma_r):
    
    padline = int((ksize-1)/2)
    
    if len(np.shape(I_o)) == 2:
        channel = 1
        (height, width) = np.shape(I_o)
        I = pad(I_o, padline, padline)
        I_bila = np.zeros((height,width))
        for h in range(height):
            for w in range(width):
                I_cropped = I[h:h+ksize, w:w+ksize]
                illu_c = I[h+padline, w+padline]
                f_bila = np.zeros((ksize,ksize))
                for i in range(ksize):
                    for j in range(ksize):
                        d_d = np.linalg.norm(padline-np.array([i,j]))
                        d_r = np.linalg.norm(illu_c-I_cropped[i,j])
                        f_bila[i][j] = MyGaussian(d_d,sigma_d)*MyGaussian(d_r, sigma_r)
                f_bila_n = f_bila/np.sum(f_bila)
                I_bila[h,w] = np.sum(I_cropped*f_bila_n)
    else:
        channel = 3
        (height, width, _) = np.shape(I_o)
        I = np.zeros((height+2*padline, width+2*padline, 3))
        for ch in range(3):
            I[:,:,ch] = pad(I_o[:,:,ch], padline, padline)
            
        I_bila = np.zeros((height,width,channel))
        
        for h in range(height):
            for w in range(width):
                I_cropped = I[h:h+ksize, w:w+ksize, :]
                illu_c = I[h+padline, w+padline, :]
                f_bila = np.zeros((ksize,ksize))
                for i in range(ksize):
                    for j in range(ksize):
                        d_d = np.linalg.norm(padline-np.array([i,j]))
                        d_r = np.linalg.norm(illu_c-I_cropped[i,j,:])
                        f_bila[i][j] = MyGaussian(d_d,sigma_d)*MyGaussian(d_r, sigma_r)
                f_bila_n = f_bila/np.sum(f_bila)
                for ch in range(channel):
                    I_bila[h,w,ch] = np.sum(I_cropped[:,:,ch]*f_bila_n)
                    
    return I_bila
#------------------------------------------------------------------------------
# Answer every question if this script is being invoked as the main module.

if __name__ == '__main__':
    for key, value in list(globals().items()):
        if key.startswith('answer_'):
            value()


# In[ ]:




