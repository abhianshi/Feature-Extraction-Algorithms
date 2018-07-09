from PIL import Image
import numpy as np
from pylab import *
from scipy import *
import tensorflow as tf
import matplotlib.cm as cm
import cv2

# X-Derivative - 2D mask
def XDerivative(I):
    mask = [[-1,1],[-1,1]]
    Ix = np.zeros(I.shape)
    row,column = I.shape
    for i in range(1,row):
        for j in range(1,column):
            Ix[i][j] = I[i-1][j-1] * mask[0][0] + I[i-1][j] * mask[0][1] + I[i][j-1] * mask[1][0] + I[i][j] * mask[1][1]
    return Ix

# Y-Derivative - 2D mask
def YDerivative(I):
    mask = [[-1,-1],[1,1]]
    Iy = np.zeros(I.shape)
    row,column = I.shape
    for i in range(1,row):
        for j in range(1,column):
            Iy[i][j] = I[i-1][j-1] * mask[0][0] + I[i-1][j] * mask[0][1] + I[i][j-1] * mask[1][0] + I[i][j] * mask[1][1]
    return Iy

# Time-Derivative - 2D mask
def TDerivative(I1, I2):
    It = I2 - I1
    return It


# A two-dimensional Gaussian mask G(gaussian_mask) G(x)
def GaussianMask(sigma, size):
    mid = int(size/2)
    gaussian_mask = [[0]*size]*size
    # constant used in the Gaussian function
    const = 1/(2*pi*(sigma**2))

    for i in range(-mid,mid+1):
        for j in range(-mid,mid+1):

            # power used in the Gaussian function
            power = (-1)*(i*i + j*j)/(2*sigma*sigma)
            gaussian_mask[i+mid][j+mid] = const*np.exp(power)
    return gaussian_mask


# Smoothing image I with Gaussian filter
def convolve(I,gaussian_mask):
    size = len(gaussian_mask[0])
    mid = int(size/2)
    out = np.zeros(I.shape)
    row,column = I.shape

    for i in range(mid,row-mid):
        for j in range(mid,column-mid):

            # Smmothening every pixel in the image
            for k in range(size):
                for l in range(size):
                    out[i][j] += I[i-mid+k][j-mid+l]*gaussian_mask[k][l]
    return out

# Generating random colours
import random
def hex_code_colors():
    a = hex(random.randrange(0,256))
    b = hex(random.randrange(0,256))
    c = hex(random.randrange(0,256))
    a = a[2:]
    b = b[2:]
    c = c[2:]
    if len(a)<2:
        a = "0" + a
    if len(b)<2:
        b = "0" + b
    if len(c)<2:
        c = "0" + c
    z = a + b + c

    return "#" + z.upper()

# Lucas Kanade function with parameters as Image1, Image2 and threshold
def LucasKanadeuv(I1,I2,threshold):

    # Gaussian Mask
    sigma = 1
    size = 5
    gaussian_mask = GaussianMask(sigma,size)

    # Smoothing I1 and I2
    smooth_I1 = convolve(I1,gaussian_mask)
    smooth_I2 = convolve(I2,gaussian_mask)

    # Image Derivatives
    I1x = XDerivative(smooth_I1)
    I1y = YDerivative(smooth_I1)

    I2x = XDerivative(smooth_I2)
    I2y = YDerivative(smooth_I2)

    Ix = I1x
    Iy = I1y

    # We can also use the summmation of the derivatives of both the images, no difference in output
    # Ix = (I1x + I2x)/2
    # Iy = (I1y + I2y)/2
    It = TDerivative(smooth_I1,smooth_I2)

    # finding the good features with maximum features as 1000, quality as 0.01 and distance as 12.
    # Changing these parameters changes the number of features detected
    features = cv2.goodFeaturesToTrack(I1 ,500,0.02,12)
    features = np.int0(features)

    #Different colors
    #colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # Initialzation of u and v matrix
    u_matrix = np.zeros(Ix.shape)
    v_matrix = np.zeros(Ix.shape)
    row,column = Ix.shape

    # Consider a window size of 3
    window_size = 3
    size_2 = int(window_size/2)

    # Parsing the image
    for f in features:
        j,i = f.ravel()

        # To make the calculations simple, we used the least sqaures fit
        # We minimize the energy function by assuming constant brightness as the motion is very small.
        # Initialzing Tensor values
        Txx,Tyy,Txy,Txt,Tyt = (0.0,0.0,0.0,0.0,0.0)
        # Parsing the window size matrix and calculating tensor values
        for k in range(-size_2,size_2+1):
            for l in range(-size_2,size_2+1):
                Txx += Ix[i+k][j+l]*Ix[i+k][j+l]
                Tyy += Iy[i+k][j+l]*Iy[i+k][j+l]
                Txy += Ix[i+k][j+l]*Iy[i+k][j+l]
                Txt += Ix[i+k][j+l]*It[i+k][j+l]
                Tyt += Iy[i+k][j+l]*It[i+k][j+l]

        denominator = Txx * Tyy - Txy * Txy
        if(denominator != 0):
            u_matrix[i][j] = (Tyt * Txy - Txt * Tyy) / denominator
            v_matrix[i][j] = (Txt * Txy - Tyt * Txx) / denominator

        # Adding the threshold so that the features with no motion are not included in optical flow
        if(abs(u_matrix[i,j]) > threshold or abs(v_matrix[i,j]) > threshold):
            plt.arrow(j,i,v_matrix[i,j],u_matrix[i,j],head_width = 5, head_length = 5, color=hex_code_colors())

    plt.imshow(I1, cmap = 'gray')
    plt.show()
    return u_matrix,v_matrix



# Read image and store it in a matrix
img1 = Image.open("/Users/abhianshusingla/Downloads/grove1.png").convert('L')
img2 = Image.open("/Users/abhianshusingla/Downloads/grove2.png").convert('L')

# img1 = Image.open("/Users/abhianshusingla/Downloads/basketball1.png").convert('L')
# img2 = Image.open("/Users/abhianshusingla/Downloads/basketball2.png").convert('L')

I1 = array(img1)
I2 = array(img2)

threshold = 0.1
u_matrix,v_matrix = LucasKanadeuv(I1,I2,threshold)
