from PIL import Image
import numpy as np
from pylab import *
from scipy import *
import time


# Calculate the Image derivative of Image I in x-direction using derivative mask
# Backward Derivative mask is used
def backwardderivativeX(I):
    Ix = np.zeros(I.shape)
    derivativemask = [-1,1]
    for i in range(row):
        for j in range (1,column):
            Ix[i][j] = I[i][j-1]*derivativemask[0] + I[i][j]*derivativemask[1]
    return Ix


# Calculate the Image derivative of Image I in y-direction using derivative mask
# Backward Derivative mask is used
def backwardderivativeY(I):
    Iy = np.zeros(I.shape)
    derivativemask = [-1,1]
    for i in range (1,row):
        for j in range(column):
            Iy[i][j] = I[i-1][j]*derivativemask[0] + I[i][j]*derivativemask[1]
    return Iy


# Calculate the Image derivative of Image I in x-direction using derivative mask
# Center Derivative mask is used
def centerderivativeX(I):
    Ix = np.zeros(I.shape)
    derivativemask = [-1,0,1]
    for i in range(row):
        for j in range (1,column-1):
            Ix[i][j] = I[i][j-1]*derivativemask[0] + I[i][j]*derivativemask[1] + I[i][j+1]*derivativemask[2]
    return Ix


# Calculate the Image derivative of Image I in y-direction using derivative mask
# Center Derivative mask is used
def centerderivativeY(I):
    Iy = np.zeros(I.shape)
    derivativemask = [-1,0,1]
    for i in range (1,row-1):
        for j in range(column):
            Iy[i][j] = I[i-1][j]*derivativemask[0] + I[i][j]*derivativemask[1] + I[i+1][j]*derivativemask[2]
    return Iy


# Part 1
# Hessian Matrix is computed using second order Image derivative
def computeHessianMatrix(I,k):
    '''
    # In some cases, backward derivative gives better result than center derivative
    # First-order Image derivatives of I in x and y direction
    Ix = backwardderivativeX(I)
    Iy = backwardderivativeY(I)

    # Second-order Image derivatives of Ix and Iy in x and y direction
    Ixx = backwardderivativeX(Ix)
    Iyy = backwardderivativeY(Iy)
    Ixy = backwardderivativeY(Ix)
    '''
    # First-order Image derivatives of I in x and y direction
    Ix = centerderivativeX(I)
    Iy = centerderivativeY(I)

    # Second-order Image derivatives of Ix and Iy in x and y direction
    Ixx = centerderivativeX(Ix)
    Iyy = centerderivativeY(Iy)
    Ixy = centerderivativeY(Ix)

    for i in range(row):
        for j in range(column):

            # Hessian Matrix
            H = np.zeros((2,2))
            H[0][0] = Ixx[i][j]
            H[0][1] = Ixy[i][j]
            H[1][0] = Ixy[i][j]
            H[1][1] = Iyy[i][j]

            # Built-in function to calculate the eigen values
            lambda1, lambda2 = linalg.eigvals(H)
            '''
            # If determinant is 0, then many false corners are detected
            if(lambda1*lambda2 == 0):
                 output[i][j] = 255
                 # Plotting the red circle on the corners
                 circ = Circle((j,i),1, color = 'r')
                 ax.add_patch(circ)
            '''
            # Check if both the lambda values are larger enough, by checking it with k
            if(lambda1 > k and lambda2 > k):
                # Plotting the red circle on the corners
                circ = Circle((j,i),1, color = 'r')
                ax.add_patch(circ)


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
            print(gaussian_mask[i+mid][j+mid])
    return gaussian_mask


# Laplacian is obtained after smoothing image I with Gaussian filter with standard deviation > 0
def LoGMatrix(I,gaussian_mask):
    LoG = np.zeros(I.shape)
    for i in range(mid,row-mid):
        for j in range(mid,column-mid):

            # Laplacian of Gaussian for every pixel in the image
            for k in range(size):
                for l in range(size):
                    LoG[i][j] += I[i-mid+k][j-mid+l]*gaussian_mask[k][l]
    return LoG


# Part 2
# Compute Hessian matrix using Laplacian of Gaussian
def computeHessianMatrixLoG(L,alpha):
    '''
    # Backward derivative is taken
    Lx = backwardderivativeX(L)
    Ly = backwardderivativeY(L)
    '''
    # Take the first-order derivative of Image LoG obtained after smoothing the image with Gaussian mask
    Lx = centerderivativeX(L)
    Ly = centerderivativeY(L)

    for i in range(row):
        for j in range(column):

            # Hessian Matrix
            H = np.zeros((2,2))
            H[0][0] = Lx[i][j]**2
            H[0][1] = Lx[i][j]*Ly[i][j]
            H[1][0] = H[0][1]
            H[1][1] = Ly[i][j]**2

            determinant = H[0][0]*H[1][1] - H[0][1]*H[1][0]
            trace = H[0][0] + H[1][1]
            CornerMeasure = determinant - alpha*trace

            # Checking CornerMeasure values
            if(CornerMeasure < -9.5 and CornerMeasure > -10):
                # Plotting the red circles on the image
                circ = Circle((j,i),1, color = 'r')
                ax2.add_patch(circ)


# Part 3
# Compute Hessian matrix using Laplacian of Gaussian
def computeHessianMatrixLoG2(L,alpha):
    '''
    # Backward derivative is taken
    Lx = backwardderivativeX(L)
    Ly = backwardderivativeY(L)
    '''
    # Take the first-order derivative of Image LoG obtained after smoothing the image with Gaussian mask
    Lx = centerderivativeX(L)
    Ly = centerderivativeY(L)

    for i in range(row):
        for j in range(column):

            # Hessian Matrix
            H = np.zeros((2,2))
            H[0][0] = Lx[i][j]**2
            H[0][1] = Lx[i][j]*Ly[i][j]
            H[1][0] = H[0][1]
            H[1][1] = Ly[i][j]**2

            # Corner Measure
            lambda1, lambda2 = linalg.eigvals(H)
            CornerMeasure = lambda1*lambda2 - alpha*(lambda1 + lambda2)

            if(CornerMeasure < -9.5 and CornerMeasure > -10):
                # Plotting the red circles on the image
                circ = Circle((j,i),1, color = 'r')
                ax3.add_patch(circ)


# Read grey scale image and store it in a matrix after converting it into grey scale using convert function as some images have 3 values in shape function
# First two values are the rows and columns of the image and the third value is the channel, so to remove the third channel,convert is used
I = array(Image.open("/Users/abhianshusingla/Downloads/input3.png").convert('L'))
row, column = I.shape
ax = plt.subplot(2,2,1)
ax.set_title('Original Image')
ax.imshow(I,cmap = 'gray')

# Value to compare the eigen values
# k = 50 for input1
# k = 130 for input3 and input2 as this image is very dense and so many corners are detected for small values of k
k = 130

# Part 1 - Compute Hessian Matrix using second-order derivatives
# Plotiing the circles and Image on the plot
ax = plt.subplot(2,2,2)
ax.set_aspect('equal')
ax.set_title('Hessian-1')
ax.imshow(I,cmap = 'gray')

computeHessianMatrix(I,k)


# parameters used to calculate gaussian mask
sigma = 1
size = 3
mid = int(size/2)
gaussian_mask = GaussianMask(sigma,size)

# Smoothing of Image I with Gaussian Mask
LoG = LoGMatrix(I,gaussian_mask)

# Feasible alpha values 0.04-0.15
# Large value of alpha means it is less sensitive
# Small value of alpha means it more sensitive
# We used alpha as 0.04
alpha = 0.04

# Part 2 - Compute Hessian Matrix using Laplacian of Gaussian
start = time.clock()

# Plotiing the circles and Image on the plot
ax2 = plt.subplot(2,2,3)
ax2.set_aspect('equal')
ax2.set_title('Hessian-2')
ax2.imshow(I,cmap = 'gray')

computeHessianMatrixLoG(LoG,alpha)

end = time.clock()
print("Time taken by part 2 is ", (end - start))

# Part 3 - Compute Hessian Matrix using Laplacian of Gaussian with different Corner Measure calculations
# Feasible alpha values 0.04-0.15
start = time.clock()

# Plotiing the circles and Image on the plot
ax3 = plt.subplot(2,2,4)
ax3.set_aspect('equal')
ax3.set_title('Hessian-3')
ax3.imshow(I,cmap = 'gray')

computeHessianMatrixLoG2(LoG,alpha)

end = time.clock()
print("Time taken by part 3 is ", (end - start))

plt.show()

# Input1
# Time taken by part 2 is  3.771166000000001
# Time taken by part 3 is  17.533431

# Input2
# Time taken by part2 is 5.076134000000003
# Time taken by part3 is 23.764975000000007

# Input3
#Time taken by part 2 is  6.231768000000002
#Time taken by part 3 is  29.108300999999997

# Calculating eigen values from a matrix takes a long time, so part 3 is very time consuming, but the end result is same.
