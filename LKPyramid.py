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


# Lucas Kanade function with parameters as smoothed Image1, smoothed Image2 and threshold
def LucasKanadeuv(smooth_I1,smooth_I2,features):

    # Image Derivatives
    I1x = XDerivative(smooth_I1)
    I1y = YDerivative(smooth_I1)

    I2x = XDerivative(smooth_I2)
    I2y = YDerivative(smooth_I2)

    Ix = (I1x + I2x)/2
    Iy = (I1y + I2y)/2
    It = TDerivative(smooth_I1,smooth_I2)

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

    return u_matrix,v_matrix

# Utility function to reduce the Gaussian Image size by 2*2
def reduce(image_l1,weight_2D):
    row = image_l1.shape[0] / 2
    column = image_l1.shape[1] / 2
    image_l0 = np.zeros((row,column))
    for i in range(2,row-2):
        for j in range(2,column-2):
            image_l0[i][j] = np.sum(np.multiply(weight_2D,image_l1[2*i-2:2*i+3,2*j-2:2*j+3]))
    return image_l0


# Utility function to expand the Gaussian Image size by 2*2
def expand(image_l1,weight_2D):
    row = image_l1.shape[0] * 2
    column = image_l1.shape[1] * 2
    image_l0 = np.zeros((row,column))
    for i in range(2,row-2):
        for j in range(2,column-2):
            for k in range(-2,3):
                for l in range(-2,3):
                    if(((i-k)/2)*2 == i-k ):
                        image_l0[i][j] += weight_2D[k+2][l+2] * image_l1[(i-k)/2][(j-l)/2]
    return image_l0


# Bilinear Interpolation
def interpolate(img):
    row = img.shape[0]
    column = img.shape[1]
    img_star = np.zeros((row*2,column*2))
    for i in range(row):
        for j in range(column):
            img_star[2*i][2*j] = img[i][j]
    return img_star


# Lucas Kanade Pyramid function with different levels of pyramid
# Algorithm:
# 1. Compute simple Lucas Kanade at highes level
# 2. At level i:
#     - Take flow u[i-1] and v[i-1] at level i-1
#     - Bilinearly interpolate it to create u_star and v_star matrices of twice resolution for level i
#     - Multiply u_star and v_star by 2
#     - Compute the Derivatives from the block displaced by u_star and v_star
#     - Apply Lucas Kanade to get corrections in the flow
#     - Add the corrections for actual optical flow
def LKPyramid(vector_u,vector_v,level):

    # Going down from highest level to lowest level in a pyramid
    for i in range(level-2,-1,-1):

        # Take the optical flow at layer i + 1 and bilinearly interpolate it for level i
        u_star = interpolate(vector_u[0]) * 2
        v_star = interpolate(vector_v[0]) * 2

        # Now the frame 2 is taken by adding the optical flows to the original image for displacement
        # Shifted Second frame w.r.t first one
        frame2 = np.zeros(gaussian_pyramid_I2[i].shape)
        for r in range(gaussian_pyramid_I2[i].shape[0]):
            for c in range(gaussian_pyramid_I2[i].shape[1]):
                frame2[r][c] = gaussian_pyramid_I2[i][r + int(u_star[r][c])][c + int(v_star[r][c])]

        # Corrections
        u_bar,v_bar = LucasKanadeuv(gaussian_pyramid_I1[i],frame2,features)

        # Correcting the optical flow at level i by adding correction u_bar and v_bar to u_star and v_star
        u_correct = u_star + u_bar
        v_correct = v_star + v_bar
        vector_u.insert(0,u_correct)
        vector_v.insert(0,v_correct)
    return vector_u,vector_v



# Read image and store it in a matrix
# img1 = Image.open("/Users/abhianshusingla/Downloads/grove1.png").convert('L')
# img2 = Image.open("/Users/abhianshusingla/Downloads/grove2.png").convert('L')

img1 = Image.open("/Users/abhianshusingla/Downloads/basketball1.png").convert('L')
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
level = 4
#weight_1D = [0.05,0.25,0.40,0.25,0.05]
weight_2D = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]) / 256.0

# Initialization of Gaussian Pyramids of Image1 and Image2
gaussian_pyramid_I1 = []
gaussian_pyramid_I1.append(smooth_I1)
gaussian_pyramid_I2 = []
gaussian_pyramid_I2.append(smooth_I2)

# Making the gaussian pyramids of Image1 and Image2 and also storing the optical flow vectors u and v for eaxh level
for i in range(1,level):
    gaussian_pyramid_I1.append(reduce(gaussian_pyramid_I1[i-1],weight_2D))
    gaussian_pyramid_I2.append(reduce(gaussian_pyramid_I2[i-1],weight_2D))


gaussian_pyramid_I1[level-1] = np.float32(gaussian_pyramid_I1[level-1])

# finding the good features with maximum features as 1000, quality as 0.01 and distance as 12.
# Changing these parameters changes the number of features detected
features = cv2.goodFeaturesToTrack(gaussian_pyramid_I1[level-1],1000,0.02,12)
features = np.int0(features)

# Calling the normal Lucas Kanade function for the lowest level, i.e. for actual Images
threshold = 0
u_matrix,v_matrix = LucasKanadeuv(gaussian_pyramid_I1[level-1],gaussian_pyramid_I2[level-1],features)

vector_u = []
vector_v = []
vector_u.append(u_matrix)
vector_v.append(v_matrix)

# Optical flow vectors at the each level
vector_u,vector_v = LKPyramid(vector_u,vector_v,level)


# Plotting the optical flows with values above threshold at each level
for k in range(level):
    num = pow(2,level-k-1)
    ax = plt.subplot(1,level,k+1)
    for f in features:
        j,i = f.ravel()
        # Can plot arrows for only values greater than threshold whose motion is negligible(corner points but no motion)
        #if(abs(vector_u[k][i][j]) > threshold or abs(vector_u[k][i][j]) > threshold ):
        ax.arrow(num*j,num*i,vector_v[k][i][j],vector_u[k][i][j],head_width = 3 * (level - k), head_length = 3 * (level - k), color=hex_code_colors())
    ax.imshow(gaussian_pyramid_I1[k], cmap = 'gray')

plt.show()
