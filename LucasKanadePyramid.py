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
    out = np.zeros(I.shape)
    row,column = I.shape

    for i in range(mid,row-mid):
        for j in range(mid,column-mid):

            # Smmothening every pixel in the image
            for k in range(size):
                for l in range(size):
                    out[i][j] += I[i-mid+k][j-mid+l]*gaussian_mask[k][l]
    return out


# Consider a window size of 3
def LucasKanadeuv(I1,Ix,Iy,It,threshold):
    # finding the good features
    features = cv2.goodFeaturesToTrack(I1 ,10000,0.01,10)
    features = np.int0(features)
    arrow_size = Ix.shape[0]/120
    # Initialzation of u and v matrix
    u_matrix = np.zeros(Ix.shape)
    v_matrix = np.zeros(Ix.shape)
    row,column = Ix.shape
    window_size = 3
    size_2 = int(window_size/2)

    # Parsing the image
    for f in features:
        j,i = f.ravel()

        # Initialzing Tensor values
        Txx,Tyy,Txy,Txt,Tyt = (0,0,0,0,0)
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
            #print('new features',u_matrix[i][j],v_matrix[i][j])

        if(abs(u_matrix[i,j]) > threshold or abs(v_matrix[i,j]) > threshold):
            plt.arrow(j,i,v_matrix[i,j],u_matrix[i,j],head_width = arrow_size, head_length = arrow_size, color='r')
            #color = cm.colormap(norm(arctan2(u_matrix[i][j], v_matrix[i][j])))

    plt.imshow(I1, cmap = 'gray')
    return u_matrix,v_matrix


def reduce(image_l1,weight_2D):
    row = image_l1.shape[0] / 2
    column = image_l1.shape[1] / 2
    image_l0 = np.zeros((row,column))
    for i in range(2,row-2):
        for j in range(2,column-2):
            image_l0[i][j] = np.sum(np.multiply(weight_2D,image_l1[2*i-2:2*i+3,2*j-2:2*j+3]))
    return image_l0


def expand(image_l1,weight_2D):
    row = image_l1.shape[0] * 2
    column = image_l1.shape[1] * 2
    print(row,column)
    image_l0 = np.zeros((row,column))
    for i in range(2,row-2):
        for j in range(2,column-2):
            for k in range(-2,3):
                for l in range(-2,3):
                    if(((i-k)/2)*2 == i-k ):
                        image_l0[i][j] += weight_2D[k+2][l+2] * image_l1[(i-k)/2][(j-l)/2]
    return image_l0




# Read image and store it in a matrix
# img1 = Image.open("/Users/abhianshusingla/Downloads/grove1.png").convert('L')
# img2 = Image.open("/Users/abhianshusingla/Downloads/grove2.png").convert('L')

img1 = Image.open("/Users/abhianshusingla/Downloads/basketball2.png").convert('L')
img2 = Image.open("/Users/abhianshusingla/Downloads/basketball2.png").convert('L')

I1 = array(img1)
I2 = array(img2)

# Gaussian Mask
sigma = 1
size = 5
mid = int(size/2)
gaussian_mask = GaussianMask(sigma,size)

# Smoothing I1 and I2
smooth_I1 = convolve(I1,gaussian_mask)
smooth_I2 = convolve(I2,gaussian_mask)

# Pyramids
level = 3
#weight_1D = [0.05,0.25,0.40,0.25,0.05]
weight_2D = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]) / 256.0

gaussian_pyramid_I1 = []
gaussian_pyramid_I1.append(smooth_I1)
ax = plt.subplot(2,level,1)
ax.imshow(gaussian_pyramid_I1[0],cmap = 'gray')
for i in range(1,level):
    gaussian_pyramid_I1.append(reduce(gaussian_pyramid_I1[i-1],weight_2D))
    ax = plt.subplot(2,level,i+1)
    ax.set_aspect('equal')
    ax.imshow(gaussian_pyramid_I1[i],cmap = 'gray')


threshold = -0.5
u_matrix,v_matrix = LucasKanadeuv(I1,Ix,Iy,It,threshold)

'''
gaussian_pyramid__expand_I1 = []
gaussian_pyramid__expand_I1.append(gaussian_pyramid_I1[2])
ax = plt.subplot(2,level,level+1)
ax.imshow(gaussian_pyramid__expand_I1[0],cmap='gray')
for i in range(1,level):
    gaussian_pyramid__expand_I1.append(expand(gaussian_pyramid__expand_I1[i-1],weight_2D))
    ax = plt.subplot(2,level,level+i+1)
    ax.imshow(gaussian_pyramid__expand_I1[i],cmap = 'gray')
'''


'''
# Image Derivatives
I1x = XDerivative(smooth_I1)
I1y = YDerivative(smooth_I1)

I2x = XDerivative(smooth_I2)
I2y = YDerivative(smooth_I2)

Ix = (I1x + I2x)/2
Iy = (I1y + I2y)/2
It = TDerivative(smooth_I1,smooth_I2)


# finding the good features
features = cv2.goodFeaturesToTrack(I1 ,10000,0.01,10)
features = np.int0(features)

threshold = -0.5
u_matrix,v_matrix = LucasKanadeuv(I1,Ix,Iy,It,features,threshold)
'''
plt.show()
