from PIL import Image
import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2 as cv

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window over which we take pool
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))

def compare():

    kernel = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])

    image = Image.open('alef_basic.png').convert('L')
    image = image.resize((34,34), resample=Image.NEAREST)
    np_image = np.array(image)
    conv1 = convolve2D(np_image, kernel)
    pool1 = pool2d(conv1, kernel_size=2, stride=2, padding=0, pool_mode='max')
    conv2 = convolve2D(pool1, kernel)
    pool2 = pool2d(conv2, kernel_size=2, stride=2, padding=0, pool_mode='max')

    img = pool2
    plt.subplot(3,2,1)
    plt.title("Original Image")
    plt.imshow(np_image, cmap='gray')
    plt.xlabel(f'{np_image.shape}')
    plt.subplot(3,2,2)
    plt.title("Twice Pooled")
    plt.imshow(img, cmap='gray')
    plt.xlabel(f'{img.shape}')

    image = Image.open('alef_64.png').convert('L')
    np_image = np.array(image)
    conv1 = convolve2D(np_image, kernel)
    pool1 = pool2d(conv1, kernel_size=2, stride=2, padding=0, pool_mode='max')
    conv2 = convolve2D(pool1, kernel)
    pool2 = pool2d(conv2, kernel_size=2, stride=2, padding=0, pool_mode='max')

    img = pool2
    plt.subplot(3,2,3)
    plt.title("Origial Image")
    plt.imshow(np_image, cmap='gray')
    plt.xlabel(f'{np_image.shape}')
    plt.subplot(3,2,4)
    plt.title("Twice Pooled")
    plt.imshow(img, cmap='gray')
    plt.xlabel(f'{img.shape}')

    image = Image.open('alef_32.png').convert('L')
    np_image = np.array(image)
    conv1 = convolve2D(np_image, kernel)
    pool1 = pool2d(conv1, kernel_size=2, stride=2, padding=0, pool_mode='max')
    conv2 = convolve2D(pool1, kernel)
    pool2 = pool2d(conv2, kernel_size=2, stride=2, padding=0, pool_mode='max')

    img = pool2
    plt.subplot(3,2,5)
    plt.title("Origial Image")
    plt.imshow(np_image, cmap='gray')
    plt.xlabel(f'{np_image.shape}')
    plt.subplot(3,2,6)
    plt.title("Twice Pooled")
    plt.imshow(img, cmap='gray')
    plt.xlabel(f'{img.shape}')

    plt.show()

kernel = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])

image = Image.open('!test/alef1.png').convert('L')
np_image = np.array(image)
conv1 = convolve2D(np_image, kernel)
pool1 = pool2d(conv1, kernel_size=2, stride=2, padding=0, pool_mode='max')
conv2 = convolve2D(pool1, kernel)
pool2 = pool2d(conv2, kernel_size=2, stride=2, padding=0, pool_mode='max')

img = pool2
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(np_image, cmap='gray')
plt.xlabel(f'{np_image.shape}')
plt.subplot(1,2,2)
plt.title("Twice Pooled")
plt.imshow(img, cmap='gray')
plt.xlabel(f'{img.shape}')
plt.show()