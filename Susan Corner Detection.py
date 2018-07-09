from PIL import Image
import numpy as np
from pylab import *
from scipy import *


# 37 size value susan mask, but taking 2D array (7*7 = 49), marking other 12 values 0 for convenience
def SusanMask():
    mask = np.zeros((7,7))
    mid = 3
    #center pixels
    mask[mid][mid] = 1
    mask[mid-1][mid] = mask[mid+1][mid] = mask[mid][mid+1] = mask[mid][mid-1] = 1

    #1st radius pixels
    mask[mid-1][mid-1] = mask[mid-1][mid+1] = mask[mid+1][mid-1] = mask[mid+1][mid+1] = (1/2)
    mask[mid-2][mid] = mask[mid+2][mid] = mask[mid][mid+2] = mask[mid][mid-2] = (1/2)

    #2nd radius pixels
    mask[mid-3][mid] = mask[mid+3][mid] = mask[mid][mid+3] = mask[mid][mid-3] = (1/3)
    mask[mid-2][mid-1] = mask[mid-1][mid-2] = mask[mid-2][mid+1] = mask[mid-1][mid+2] = (1/3)
    mask[mid+2][mid-1] = mask[mid+1][mid-2] = mask[mid+2][mid+1] = mask[mid+1][mid+2] = (1/3)

    #3rd radius pixels
    mask[mid-3][mid-1] = mask[mid-2][mid-2] = mask[mid-1][mid-3] = (1/4)
    mask[mid-3][mid+1] = mask[mid-2][mid+2] = mask[mid-1][mid+3] = (1/4)
    mask[mid+3][mid-1] = mask[mid+2][mid-2] = mask[mid+1][mid-3] = (1/4)
    mask[mid+3][mid+1] = mask[mid+2][mid+2] = mask[mid+1][mid+3] = (1/4)

    return mask


# Creating USAN matrix
def USANMatrix(I,mask,threshold):
    N = np.zeros(I.shape)
    for i in range(3,row-3):
        for j in range(3,column-3):

            # Center value (nucleus), so circular mask is 1 for center value
            Io = I[i][j]

            # Calculate the number of pixels of USAN n(r0) at location (i,j)
            for k in range(7):
                for l in range(7):
                    if(mask[k][l] != 0):
                        Ir = mask[k][l]*I[i-3+k][j-3+l]
                        power = (-1)*(((Ir-Io)/threshold)**6)
                        N[i][j] += (np.exp(power))
    return N


# Calculate the index for which N has the maximum value
# Value of g is half of this maximum value
# g = 37/2
def calcG(N):
    g = (N.max())/2
    return g


# Computing strength of corner
def strength(N,g):
    R = np.zeros(N.shape)
    for i in range(row):
        for j in range(column):
            if(N[i][j] <= g):
                R[i][j] = g - N[i][j]
                #print(R[i][j])
    return R


# Computing non-maximum suppression by taking into consideration the immediate neighbours
def suppression(R,ax):
    for i in range(3,row-3):
        for j in range(3,column-3):
            if((R[i][j] > R[i][j-1]) and (R[i][j] > R[i][j+1]) and (R[i][j] > R[i-1][j]) and (R[i][j] > R[i-1][j-1]) and (R[i][j] > R[i-1][j+1]) and (R[i][j] > R[i+1][j]) and (R[i][j] > R[i+1][j-1]) and (R[i][j] > R[i+1][j+1])):
                circ = Circle((j,i),1, color = 'r')
                ax.add_patch(circ)


# Computing non-maximum suppression taking in consideration the size of SUSAN mask, i.e., 37
def suppression2(R,ax):
    for i in range(3,row-3):
        for j in range(3,column-3):

            isCorner = True
            for k in range(7):
                for l in range(7):

                    if(k != 3 or l != 3):
                        if(R[i][j] <= R[i-3+k][j-3+l]):
                            isCorner = False
            if(isCorner):
                # Plotting the red circle on the corners
                circ = Circle((j,i),1, color = 'r')
                ax.add_patch(circ)


# Calculate centroid of suppressed images
def centroid(I):
    C = np.zeros(I.shape)
    for i in range(3,row-3):
        for j in range(3,column-3):

            for k in range(7):
                for l in range(7):
                    C[i][j] += I[i-3+k][j-3+l]

            C[i][j] = C[i][j]/37
    return C


# Calculate distance of nucleus from the centroid
def distcentroid(S,C,k):
    D = np.zeros(I.shape)
    for i in range(3,row-3):
        for j in range(3,column-3):
            if(abs(C[i][j] - S[i][j]) > k):
                D[i][j] = 255
                #print(S[i][j], C[i][j], C[i][j] - S[i][j])
    return D


# Denoising
# Applying median filter to remove noise
def medianFilter(I,medsize):
    foutput = np.zeros(I.shape)

    # Creating a 2D median filter
    filter = np.zeros((medsize*2+1,medsize*2+1))
    for i in range(medsize,row-medsize):
        for j in range(medsize, column - medsize):

            # Assigning the median value of the flattened sorted array in the median filter
            for k in range(2*medsize+1):

                # Taking the elements row-wise
                filter[k] = I[i+k-medsize][j+k-medsize:j+k-medsize+1]

            filter = filter.flatten()
            filter.sort()
            foutput[i][j] = filter[medsize]
    return foutput


# Smoothing
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


# Convolve the image I with gaussian mask
def convolve(I,gaussian_mask):
    Ix = np.zeros(I.shape)
    for i in range(mid,row-mid):
        for j in range(mid,column-mid):

            # Convolve with Gaussian for every pixel in the image
            for k in range(size):
                for l in range(size):
                    Ix[i][j] += I[i-mid+k][j-mid+l]*gaussian_mask[k][l]
    return Ix


# Normalization of Image
def norm(I):
    normed = np.zeros(I.shape)
    maximum = I.max()
    normed = (I*255)/maximum
    return normed


# Run the algorithm for threshold values - 20, 30, 50, 60, 70, 80 and 100
# With 20 and 30 value, some of the corners moving from light to dark are missed
# And, with 70, 80 and 100, some of the corners moving from light dark to extremely dark are missed
# With threshold value of 50, almost all the corners are detected except for the dark to extremely dark movement
# So, for the given image, threshold value around 50 and 60 gives the best result
threshold = 60
Smask = SusanMask()

# part 1 - Normal image without noise
I = array(Image.open("/Users/abhianshusingla/Downloads/susan_input1.png"))
row, column = I.shape

N = USANMatrix(I, Smask, threshold)
g = calcG(N)
R = strength(N,g)

# Showing the image
ax1 = plt.subplot(2,2,1)
ax1.set_title('SUSAN on Original Image')
ax1.imshow(I,cmap = 'gray')

suppression2(R,ax1)

# Some images require these two extra steps in SUSAN for corner detection
#C = centroid(S)
#D = distcentroid(S,C,245)


# part 2 - SUSAN with noisy image
I2 = array(Image.open("/Users/abhianshusingla/Downloads/susan_input2.png"))

# Showing Noisy Image
ax2 = plt.subplot(2,2,2)
ax2.set_title('Noisy Image')
ax2.imshow(I2,cmap = 'gray')

threshold = 10
Smask = SusanMask()

N2 = USANMatrix(I2,Smask, threshold)
g = 1.5
R2 = strength(N2,g)

# Showing SUSAN on Noisy Image
ax3 = plt.subplot(2,2,3)
ax3.set_title('SUSAN on Noisy Image')
ax3.imshow(I2,cmap = 'gray')

suppression2(R2,ax3)


# part 3 - SUSAN after applying median filter to noisy image
# Not much difference with median size 3 and 5, so take any size
# Parameters
medsize = 5
sigma = 0.35
size = 5
mid = int(size/2)
gaussian_mask = GaussianMask(sigma,size)

# Denoising of Image with median filter
F = medianFilter(I2,medsize)

# Smoothing of denoised Image with Gaussian Mask
smooth = convolve(F,gaussian_mask)

# Normalization of smoothed Image
normed = norm(smooth)

# Susan Mask
threshold = 10
Smask = SusanMask()

# SUSAN after denoising, smoothing and normalization
N3 = USANMatrix(normed,Smask, threshold)
g = 10
R3 = strength(N3,g)

# Showing SUSAN on filtered image
ax4 = plt.subplot(2,2,4)
ax4.set_title('SUSAN on Filtered Image')
ax4.imshow(normed,cmap = 'gray')

suppression2(R3,ax4)

plt.show()

# The value of g determines the minimum size of the univalue segment.
# So, if the value of g is very large, then it will detect edges
# We started with the half of the maximum value of USAN matrix N and found out that it works best for the non-noisy image
# For the noisy images, we take the g values between 3 and 10 for the better results
# Before applying filter on the noisy image, SUSAN image is full of false corners because of salt and pepper noise
# But, an idea of the image objects is obtained.
# After applying the filter on the noisy image, egges got blurred
# After SUSAN, false corners are removed but instead of sharp corners, we get like points on the edges because of blurring or smoothing

'''
Image.fromarray(I).show()
Image.fromarray(N).show()
Image.fromarray(R).show()
Image.fromarray(C).show()
Image.fromarray(D).show()
Image.fromarray(S).show()
Image.fromarray(S2).show()
'''
