#libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sys import exit

#Function definitions begin
"""
1. Converts an RGB image to grayscale
2. & crops it, to ensure that the image's dimensions are powers of 2
3. returns a matrix
"""
def preprocessing(picture):

    pic = picture.copy()
    #Converts RGB image to  gray-scale
    pic = pic.convert('L')

    x, y = pic.size

    x_ = 2 ** np.floor( np.log2( x ) )
    y_ = 2 ** np.floor( np.log2( y ) )

    d1 = x - x_
    d2 = y - y_


    #If not, crops it
    if d1 or d2:
        pic = pic.crop((0, 0, x-d1, y-d2))

    #converts it to an array
    img = np.array(pic).astype('int32')
    del pic

    return img


#Performs Haar's transformation on a matrix
def haar(img):

    #init
    row, col = img.shape
    sqrt2 = round(np.sqrt(2), 4)

    hc = round(col/2)
    hr = round(row/2)

    #Horizontal difference
    c_sum = np.zeros((row, hc))
    c_diff = np.zeros((row, hc))

    for j in range(0, col, 2):
        c_sum[:, j//2] = (img[:, j] + img[:, j+1]) /sqrt2
        c_diff[:, j//2] = (img[:, j] - img[:, j+1]) /sqrt2

    temp = np.hstack((c_sum, c_diff))

    #plt.imshow(temp, cmap= 'gray')
    #plt.title('Check')

    #Vertical difference
    r_sum = np.zeros((hr, col))
    r_diff = np.zeros((hr, col))

    for i in range(0, row, 2):
        r_sum[i//2, :] = (temp[i, :] + temp[i+1, :]) /sqrt2
        r_diff[i//2, :] = (temp[i, :] - temp[i+1, :]) /sqrt2

    #result
    transformed = np.vstack((r_sum, r_diff))

    #clean up
    del r_sum, r_diff, c_sum, c_diff, temp
    del row, col, hr, hc, sqrt2

    return transformed.round()


#Peforms inverse of Haar's transformation
def inv_haar(img):

    #init
    row, col = img.shape
    sqrt2 = round(np.sqrt(2), 4)

    hc = round(col/2)
    hr = round(row/2)

    #Vertical reconstruction
    matrix = np.zeros((row, col))

    for i in range(0, row, 2):
        idx = i // 2
        matrix[i, :] = (img[idx, :] + img[hr + idx, :]) / sqrt2
        matrix[i+1, :] = (img[idx, :] - img[hr + idx, :]) / sqrt2

    #check
    #plt.imshow(matrix, cmap= 'gray')
    #plt.title('Check')

    #Horizontal reconstruction
    matrix2 = matrix.copy()

    for j in range(0, col, 2):
        jdx = j // 2
        matrix2[:, j] = (matrix[:, jdx] + matrix[:, hc + jdx]) / sqrt2
        matrix2[:, j+1] = (matrix[:, jdx] - matrix[:, hc + jdx]) / sqrt2

    #clean up
    del row, col, sqrt2, matrix

    #result
    return matrix2.round()


#Performs n-level Haar's transformation
#returns the answer
def dwt(pic, n):

    #init
    img = preprocessing(pic)
    row, col = img.shape

    #Performs the haar transform n times
    divisor = int(2 ** (n-1) ) + 1
    for i in range(1, divisor):

        #To avoid re-computations
        div = int(2 ** np.ceil( np.log2(i) ))
        if i != div:
            #print('doing nothing, because i != divisor:', i, divisor)
            continue

        img[:row//div, :col//div] = haar(img[:row//div, :col//div])

    #clean up
    del divisor, row, col

    return img


#Performs n-level inverse Haar's transform
def idwt(img, n):

    row, col = img.shape
    #Don't mess the original image
    temp = img.copy()

    #Performs the haar transform n times
    divisor = int(2 ** (n-1) ) + 1
    for i in range(1, divisor):

        #To reverse the loop
        den = divisor - i
        #To avoid re-computations
        div = int(2 ** np.ceil( np.log2(den) ))
        if den != div:
            #print('doing nothing, because i != divisor:', i, divisor)
            continue

        temp[:row//den, :col//den] =        inv_haar(temp[:row//den, :col//den])

    del divisor, row, col

    return temp


#Calculates relative error, between the original & recontructed matrices by using the Frobenius norm
def error(org, recon):
    tot = round(np.linalg.norm(org), 4)
    diff = round(np.linalg.norm(org - recon), 4)
    ans = round(diff/tot, 4) *100
    print("\tError:", ans, "%")


#Increases contrast in the 'detail matrices' of '1-level dwt'
#part: h, v, d = 1, 2, 3
def sharpen(inter, part):
    row, col = inter.shape
    hr, hc = row//2, col//2

    if part == 1:
        avg  = inter[:hr, hc:].mean()
        max_ = inter[:hr, hc:].max()
        min_ = inter[:hr, hc:].min()

        inter[:hr, hc:][ inter[:hr, hc:] >= avg ] = 2*max_
        inter[:hr, hc:][ inter[:hr, hc:] < avg ] = 2*min_

    if part == 2:
        avg  = inter[hr:, :hc].mean()
        max_ = inter[hr:, :hc].max()
        min_ = inter[hr:, :hc].min()

        inter[hr:, :hc][ inter[hr:, :hc] >= avg ] = 2*max_
        inter[hr:, :hc][ inter[hr:, :hc] < avg ] = 2*min_

    if part == 3:
        avg  = inter[hr:, hc:].mean()
        max_ = inter[hr:, hc:].max()
        min_ = inter[hr:, hc:].min()

        inter[hr:, hc:][ inter[hr:, hc:] >= avg ] = 2*max_
        inter[hr:, hc:][ inter[hr:, hc:] < avg ] = 2*min_

    del row, col, hr, hc


#"Truncates" a specific detail sub-matrices in an 'n-level DWT'
#H, V, D = 1, 2, 3
def sharpen_n(img, target, level):

    row, col = img.shape

    if target > 3 or target < 1:
        exit("Invalid entry in sharpenn")


    divisor = int(2 ** (level-1) ) + 1

    #H, V, D
    if target:
            for i in range(1, divisor):
                #To avoid re-computations
                div = int(2 ** np.ceil( np.log2(i) ))
                if i != div:
                #print('doing nothing, because i != divisor:', i, divisor)
                    continue

                sharpen(img[:row//i, :col//i], target)

    del row, col

def sharpen_all(inter, level):
    sharpen_n(inter, 1, level)
    sharpen_n(inter, 2, level)
    sharpen_n(inter, 3, level)

#Saves a matrix, as an image in the current directory
def save(mat, name):
    i8 = (((mat - mat.min()) / (mat.max() - mat.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(i8)
    img.save(name)

#"Truncates" a specific sub-matrix
#target: A, H, V, D  = 0, 1, 2, 3
def zerofy(img, target):

    r, c = img.shape
    hr, hc = r//2, c//2
    temp = img.copy()

    #D
    if target == 3:
        img[hr:, hc:] = np.zeros((hr, hc))
    #H
    elif target == 1:
        img[:hr, hc:] = np.zeros((hr, hc))
    #V
    elif target == 2:
        img[hr:, :hc] = np.zeros((hr, hc))
    #A
    elif target == 0:
        img[:hr, :hc] = np.zeros((hr, hc))
    else:
        exit('Invalid target')

    del temp, hr, hc, r, c

#"Truncates" a specific detail sub-matrices in an 'n-level DWT'
#A, H, V, D = 0, 1, 2, 3
def zerofyn(img, target, level):

    row, col = img.shape

    if target > 3 or target < 0:
        exit("Invalid entry in zerofyn")


    divisor = int(2 ** (level-1) ) + 1

    #H, V, D
    if target:
            for i in range(1, divisor):
                #To avoid re-computations
                div = int(2 ** np.ceil( np.log2(i) ))
                if i != div:
                #print('doing nothing, because i != divisor:', i, divisor)
                    continue

                zerofy(img[:row//i, :col//i], target)
    #A
    else:
        zerofy(img[:row//(divisor-1), :col//(divisor-1)], 0)

    del row, col

#Removes all detail soefficients, in a '3 level dwt'
def compressn(inter, level):
    zerofyn(inter, 1, level)
    zerofyn(inter, 2, level)
    zerofyn(inter, 3, level)


#------------------MAIN-------------------------#

#reads an image
pic = Image.open('img.jpg')

#Set a level for DWT
level = 1

#Performs DWT
inter = dwt(pic, level)
save(inter,"2. DWT.jpg")

#Performs inverse DWT
recon = idwt(inter, level)
save(recon,"4. Exact reconstruction.jpg")

#Increases contrast in the image
sharpen_all(inter, level)
save(inter,"3. Enhanced.jpg")


#Crops the image suitably
org = preprocessing(pic)
save(org,"1. Image.jpg")

#Calculates the error in the exact reconstruction
print("Exact reconstruction:")
error(org, recon)

#Performs compression
compressn(inter, level)
del recon
recon = idwt(inter, level)
save(recon,"5. Compressed.jpg")

#Calculates the error in the exact reconstruction
print("After compression:")
error(org, recon)

#Signifies successful execution of this entire program
print("Successful execution. Results loaded in current directory.")

#clean up
del pic, inter, recon, org
