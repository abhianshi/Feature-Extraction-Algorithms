from PIL import Image
import numpy as np
from pylab import *
from scipy import *


# A one-dimensional Gaussian mask G(gaussian_mask) G(x) = e**(-(x**2)/(2*(sigma**2)))
def GaussianMask(sigma, size):
    gaussian_mask = [0]*size

    # constant used in the Gaussian function
    const = (1/(sqrt(2*pi)*sigma))

    # For the middle element of the gaussian, that is x = 0, exponential power is 0.
    # So, gaussian mask is just the constant for middle element
    gaussian_mask[mid] = const

    # Assigning the values to both left and right of middle element. It is symmetric around the middle element.
    for i in range(mid):
        gaussian_mask[mid-i-1] = (const)*np.exp(-((i+1)*(i+1))/(2*sigma*sigma))
        gaussian_mask[mid+i+1] = gaussian_mask[mid-i-1]
    return gaussian_mask


# A one-dimensional X derivative of Gaussian mask G(derivative_Gaussian)
# Creating the derivative in x-direction, and the same array is used in y-direction column wise
def DerivativeGaussian(sigma, size):
    derivative_Gaussian = [0]*size

    # For the middle element of the gaussian derivative, that is x = 0
    derivative_Gaussian[mid] = 0

    # Assigning the values to both left and right of middle element. It is symmetric around the middle element.
    for i in range(mid):
        derivative_Gaussian[mid-i-1] = ((i+1)/(sigma*sigma))*gaussian_mask[mid-i-1]
        derivative_Gaussian[mid+i+1] = (-1)*derivative_Gaussian[mid-i-1]
    return derivative_Gaussian


# In Convolution, there is flipping unlike correlation.
# But here, Convolution and Correlation of Gaussian functions are same because the values around diagnols are symmetric.
# Convolve the image A with B along the rows to give the x component image (Ix)
def convolveX(A,B):
    Ix = np.zeros(I.shape)
    for i in range(row):
        for j in range(column):

            # Middle value,i.e, highest weightage value
            Ix[i][j] = A[i][j]*B[mid]

            # Adding the values to both left and right of middle element. It is symmetric around the middle element.
            for k in range(mid):
                if ((j-k-1) >= 0):
                    Ix[i][j] += A[i][j-k-1]*B[mid-k-1]
                if ((j+k+1) < column):
                    Ix[i][j] += A[i][j+k+1]*B[mid+k+1]
    return Ix


# Convolve the image A with B along the columns to give the y component image (Iy)
def convolveY(A,B):
    Iy = np.zeros(I.shape)
    for j in range(column):
        for i in range(row):

            # Middle value,i.e, highest weightage value
            Iy[i][j] = A[i][j]*B[mid]

            # Adding the values to both above and below to the middle element. It is symmetric around the middle element.
            for k in range(mid):
                if ((i-k-1) >= 0):
                    Iy[i][j] += A[i-k-1][j]*B[mid-k-1]
                if ((i+k+1) < row):
                    Iy[i][j] += A[i+k+1][j]*B[mid+k+1]
    return Iy


# Compute magnitude of the edge response by combining the x and y components.
def magnitude(A,B):
    mag = np.zeros(I.shape)
    for i in range(row):
        for j in range(column):
            mag[i][j] = sqrt(A[i][j]**2 + B[i][j]**2)
    return mag


# Implementing non-maximum suppression algorithm.
# Pixels that are not local maxima should be removed with this method.
# In other words, not all the pixels indicating strong magnitude are edges in fact.
# We need to remove false-positive edge locations from the image.
# For input2 image, horizontal and vertical edges are not getting detected because the magnitude is same.
# If we check for equality, then those edges are getting detected, but according to non_maximum_suppression, it should be local maxima.
def non_maximum_suppression(mag,Ixx,Iyy):
    S = np.zeros(I.shape)

    # Calculate the gradient orientation, i.e., angle theta in degrees
    for i in range(1,row-1):
        for j in range(1,column-1):
            if (Ixx[i][j] == 0):
                theta = int(math.degrees(math.pi/2))
            else:
                theta = int(math.degrees(math.atan(Iyy[i][j]/Ixx[i][j])))

            # Checking in all eight directions with respect to the pixel
            # If angle theta is between 23 and 67 degrees, then it is rouded off to 45 degrees and the diagnol2 elements are checked
            # If angle theta is between -23 and -67 degrees, then it is rouded off to -45 degrees and the diagnol1 elements are checked
            # Similarly, If angle theta is between 68 and 90 or -68 and -90 degrees, then it is rouded off to 90 degrees and the vertical elements are checked
            '''
            round off theta values
            45  - 23 -> 67 and
            90  - 68 -> 90 and (-68) -> (-90)
            -45 - (-23) -> (-67)
            0   - (-22) -> 22
            '''
            if (theta >= 23 and theta <= 67):
                if((mag[i][j] > mag[i+1][j+1]) and mag[i][j] > mag[i-1][j-1]):
                    S[i][j] = mag[i][j]
                else:
                    S[i][j] = 0
            elif ((theta >= 68 and theta <= 90) or (theta <= -68 and theta >= -90)):
                if((mag[i][j] > mag[i][j+1]) and mag[i][j] > mag[i][j-1]):
                    S[i][j] = mag[i][j]
                    #print(theta,S[i][j])
                else:
                    S[i][j] = 0
            elif (theta >= -67 and theta <= -23):
                if((mag[i][j] > mag[i-1][j+1]) and mag[i][j] > mag[i+1][j-1]):
                    S[i][j] = mag[i][j]
                else:
                    S[i][j] = 0
            elif (theta >= -22 and theta <= 22):
                if((mag[i][j] > mag[i+1][j]) and mag[i][j] > mag[i-1][j]):
                    S[i][j] = mag[i][j]
                    #print(theta,S[i][j])
                else:
                    S[i][j] = 0
    return S

# def non_maximum_suppression(mag,Ixx,Iyy):
#     S = np.zeros(I.shape)
#
#     # Calculate the gradient orientation, i.e., angle theta in degrees
#     for i in range(1,row-1):
#         for j in range(1,column-1):
#             if (Ixx[i][j] == 0):
#                 theta = int(math.degrees(math.pi/2))
#             else:
#                 theta = int(math.degrees(math.atan(Iyy[i][j]/Ixx[i][j])))
#
#             # Checking in all eight directions with respect to the pixel
#             # If angle theta is between 23 and 67 degrees, then it is rouded off to 45 degrees and the diagnol2 elements are checked
#             # If angle theta is between -23 and -67 degrees, then it is rouded off to -45 degrees and the diagnol1 elements are checked
#             # Similarly, If angle theta is between 68 and 90 or -68 and -90 degrees, then it is rouded off to 90 degrees and the vertical elements are checked
#             '''
#             round off theta values
#             45  - 23 -> 67 and
#             90  - 68 -> 90 and (-68) -> (-90)
#             -45 - (-23) -> (-67)
#             0   - (-22) -> 22
#             '''
#             if (theta >= 23 and theta <= 67):
#                 if((mag[i][j] > mag[i-1][j+1]) and mag[i][j] > mag[i+1][j-1]):
#                     S[i][j] = mag[i][j]
#                 else:
#                     S[i][j] = 0
#             elif ((theta >= 68 and theta <= 90) or (theta <= -68 and theta >= -90)):
#                 if((mag[i][j] > mag[i+1][j]) and mag[i][j] > mag[i-1][j]):
#                     S[i][j] = mag[i][j]
#                     #print(theta,S[i][j])
#                 else:
#                     S[i][j] = 0
#             elif (theta >= -67 and theta <= -23):
#                 if((mag[i][j] > mag[i+1][j+1]) and mag[i][j] > mag[i-1][j-1]):
#                     S[i][j] = mag[i][j]
#                 else:
#                     S[i][j] = 0
#
#             elif (theta >= -22 and theta <= 22):
#                 if((mag[i][j] > mag[i][j+1]) and mag[i][j] > mag[i][j-1]):
#                     S[i][j] = mag[i][j]
#                     #print(theta,S[i][j])
#                 else:
#                     S[i][j] = 0
#
#     return S


# Calculating the Hysteresis Thresholding values by considering the values in all eight directions but just the neighbourhood values.
# If any of the immediate neighbourhood is having the hysteresis Threshold value above high, then it is also marked as high
# Time Complexity of this function is low, so using this function
def hysteresisThreshold(S,low,high):
    H = np.zeros(I.shape)
    for i in range(row):
        for j in range(column):
            if(S[i][j] >= high):
                H[i][j] = 255
            elif(S[i][j] > low and S[i][j] < high):
                if(((i-1) >= 0 and H[i-1][j] == 255) or ((i-1) >= 0 and (j-1) >= 0 and H[i-1][j-1] == 255) or ((i-1) >= 0 and (j+1) < column and H[i-1][j+1] == 255) or ((i+1) < row and H[i+1][j] == 255) or ((i+1) < row and (j-1) >= 0 and H[i-1][j-1] == 255) or ((i+1) < row and (j+1) < column and H[i-1][j+1] == 255) or ((j-1) >= 0 and H[i][j-1] == 255) or ((j+1) < column and H[i][j+1] == 255)):
                    H[i][j] = 255
    return H


# constants used in the program

# size used for gaussian is - 3,5,7,11
# From the results, we can see that with increase in gaussian size matrix, edges are becoming fine and more neighbourhood values are taken into consideration
# High size means larger and smoother edges are detected - more blurring
# Low size means very fine and small edges are detected - less blurring
# Gaussian Size doesn't affect the image much as sigma and threshold values
size = 5
mid = int(size/2)

# sigma values used are - 1,2, 1.4
# With sigma value 1, most of the output seems okay
# With sigma value 2, most of the output is black and very few edges are detected
# With sigma value 1.2, almost all of the output are fine and clear.
# With increase in sigma value, less edges are detected
sigma = 1.2

# Hysteresis Thresholding low and high values
# If we increase the low value, then most small and irrelevant edges are skipped, which is a good thing but we can't bring it closer to high value as it will then take all the values between high and low as high
# And greater the high value means only significant edges are taken,
# In case of dense pictures, choose the high value really high as it will help in skipping the less dense edges
# Basically, too high values sometimes misses the important information and too less value identifies the false edges
# So, we can't choose a specific low and high for all images.
# Tree - low = 2, high = 8, sigma = 1.2
# Cruise - low =2, high = 10, sigma = 1.2
# Animals - low = 2 and high = 12, sigma = 1
low = 2
high = 12


# Read grey scale image and store it in matrix I
I = array(Image.open("/Users/abhianshusingla/Downloads/Computer Vision/Programmin Assignment 1/input2.png").convert('L'))
row, column = I.shape

# Calling Gaussian Mask function
gaussian_mask = GaussianMask(sigma,size)

# Calling derivative of Gaussian. Creating 1D array in x-direction, but using it as column wise array in y-direction
derivative_Gaussian = DerivativeGaussian(sigma,size)

# Convolve Image I with gaussian_mask to give Ix and Iy
Ix = convolveX(I,gaussian_mask)
Iy = convolveY(I,gaussian_mask)

# Convolve Image Ix with gaussian_derivative to give Ixx and Iyy
Ixx = convolveX(Ix,derivative_Gaussian)
Iyy = convolveY(Iy,derivative_Gaussian)

# Compute magnitude of the edge response(magnitude of Ixx and Iyy)
mag = magnitude(Ixx,Iyy)

# Compute non maximum suppression to remove the high magnitude false edges
S = non_maximum_suppression(mag,Ixx,Iyy)

# Apply Hysteresis thresholding to obtain final edge-map.
# H = hysteresisThreshold2(S,low, high)
H = hysteresisThreshold(S,low, high)


# Showing the images Ix, Iy, Ixx, Iyy, magnitude and after threshold using imshow method
ax = plt.subplot(2,3,1)
ax.set_title('Ix')
ax.imshow(Ix,cmap = 'gray')
ax = plt.subplot(2,3,2)
ax.set_title('Iy')
ax.imshow(Iy,cmap = 'gray')


ax = plt.subplot(2,3,3)
ax.set_title('Ixx')
ax.imshow(Ixx,cmap = 'gray')
ax = plt.subplot(2,3,4)
ax.set_title('Iyy')
ax.imshow(Iyy,cmap = 'gray')

ax = plt.subplot(2,3,5)
ax.set_title('Magnitude')
ax.imshow(mag,cmap = 'gray')

ax =plt.subplot(2,3,6)
ax.set_title('Canny-Edge')
ax.imshow(H,cmap = 'gray')

plt.show()
