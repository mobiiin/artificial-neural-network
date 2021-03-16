import numpy as np
import glob
import cv2 as cv
import os
import operator
from matplotlib import pyplot as plt
import sys


# change rgb image data to binary or bipolar list
def imgtodata(img, method):
    img_list = []
    for pixel in img:
        for r, g, b in pixel:
            if r > 128:
                img_list.append(-1) if method == 'bipolar' else img_list.append(0)
            else:
                img_list.append(1)
    return img_list


# creating the dataset and output value with desired width and height
def dataset(width, height, method='bipolar', save=False):
    # output indexing: [neuron][image]
    y_out1 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])

    if method == 'bipolar':
        y_out1 = np.where(y_out1 == 0, -1, 1)

    data1 = []
    dir_path = os.getcwd() + '\\data\\'
    i = 0
    kernel = np.ones((5, 5), np.uint8)
    # reading all the images in data folder and resizing to the desired size
    for img in glob.glob(dir_path + "/*.png"):
        img = cv.imread(img)
        img = cv.erode(img, kernel, iterations=1)
        img = cv.resize(img, (width, height))
        # performing threshold to remove the middle pixel intensities
        _, thresh1 = cv.threshold(img, 130, 255, cv.THRESH_BINARY)
        if save:
            cv.imwrite("resized/data%i.png" % i, thresh1)

        # appending all image data to a list
        data1.append(imgtodata(thresh1, method))
        i += 1

    # data indexing: [image][pixel] # data.shape = 21*63
    data1 = np.array(data1)
    data1 = data1.reshape(21, width * height)

    return data1, y_out1


def train(inputs, output, num_of_neurons=7):
    # weight indexing: [neuron][pixels]
    weight = np.zeros((num_of_neurons, len(inputs[0])), dtype=int)
    # bias indexing: [neuron]
    bias1 = np.zeros(num_of_neurons, dtype=int)

    for neuron in range(num_of_neurons):
        for image in range(len(inputs)):
            weight[neuron] += inputs[image] * output[neuron][image]
            bias1[neuron] += output[neuron][image]
    return weight, bias1


# use force_find only if the ordinary method fails to recognize
def recognize(img, weight, bias1, force_find=False):
    y_in = np.zeros(7, dtype=int)
    found = False
    for neurons in range(7):
        y_in[neurons] = np.dot(img, weight[neurons]) + bias1[neurons]
        if y_in[neurons] >= 0:
            found = True
            if neurons == 0:
                print('A')
            elif neurons == 1:
                print('B')
            elif neurons == 2:
                print('C')
            elif neurons == 3:
                print('D')
            elif neurons == 4:
                print('E')
            elif neurons == 5:
                print('J')
            elif neurons == 6:
                print('K')
        elif neurons == 6 and found is False and force_find:
            ''' here I implemented an other method to make the network recognize
            the letter with the maximum value in case it didnt find any above zero'''
            index, _ = max(enumerate(y_in), key=operator.itemgetter(1))
            # print(index)
            if index == 0:
                print('A')
            elif index == 1:
                print('B')
            elif index == 2:
                print('C')
            elif index == 3:
                print('D')
            elif index == 4:
                print('E')
            elif index == 5:
                print('J')
            elif index == 6:
                print('K')
    return y_in


# just uncomment each section and run the code

'''##### Part1 Section B
# training each neuron with one character'''

# data, y_out = dataset(7, 9, 'binary')
# weights, bias = train(data, y_out)
# # you can change the input image by changing the num after data[%]
# recognize(data[8], weights, bias)

''' these neurons cannot recognize the input character correctly
cause we used binary method to train the weights 
in each case the y_in for all of the neurons yields
more than thresh(zero) leading to firing all of the output neurons
meaning the input letter could be any of the characters 
which is not acceptable'''

''' applying some noise to input image'''

# data, y_out = dataset(7, 9, 'binary', True)
# weights, bias = train(data, y_out)
# # you can change the input image by changing the num after data%.png
# img1 = cv.imread('resized/data7.png')
# img1 = cv.erode(img1, (1, 1), iterations=1)
# # img1 = cv.dilate(img1, (1, 1), iterations=1)
# cv.imwrite('test1b.png', img1)
# img1 = imgtodata(img1, 'binary')
# recognize(img1, weights, bias)

''' this part still doesnt work same as previous part
the reason as I mentioned earlier is using the binary method
in training weights. and the network still outputs all 
the letters as seen on input'''

'''##### Part1 Section C
# training all neurons with bipolar data'''

# data, y_out = dataset(7, 9, 'bipolar')
# weights, bias = train(data, y_out)
# recognize(data[10], weights, bias)

''' as we see here the networks is more capable of recognizing
which character is being fed into, although its not fully functional.
it realizes the A C K J characters but still has a hard time 
figuring out couple of letters like B and E'''

'''applying some noise to input image'''

# data, y_out = dataset(7, 9, 'bipolar', True)
# weights, bias = train(data, y_out)
# img1 = cv.imread('resized/data5.png')
# # img1 = cv.erode(img1, (1, 1), iterations=1)
# img1 = cv.dilate(img1, (1, 1), iterations=1)
# cv.imwrite('test1c.png', img1)
# img1 = imgtodata(img1, 'binary')
# recognize(img1, weights, bias)

''' applying noise to some of the input images that the network 
was able to recognize earlier, we see that some of the images 
are still recognized with noise like image num 18 and some of 
them are nolonger recognizable'''


'''##### Part 2'''
# uncomment this then proceed to the sections
# data, _ = dataset(11, 15, 'bipolar', True)


def ocr(input_letter, output_pic_dim=(60, 60)):

    y_out2, _ = dataset(output_pic_dim[0], output_pic_dim[1], 'binary')
    # y_out2 = y_out2[::3]

    weight2, bias2 = train(data, y_out2.T, output_pic_dim[0] * output_pic_dim[1])
    # y_in = recognize(input_letter, weight2, bias2, force_find)
    # if force_find is False:
    #     index = [i for i, v in enumerate(y_in >= 0) if v]
    #     if bool(index) is False:
    #         print('character not found')
    #         sys.exit()
    #     index = np.squeeze(index[-1])
    # else:
    #     index, _ = max(enumerate(y_in), key=operator.itemgetter(1))

    character = np.zeros(output_pic_dim[0] * output_pic_dim[1])
    for pixel in range(output_pic_dim[0] * output_pic_dim[1]):
        character[pixel] = np.dot(input_letter, weight2[pixel]) + bias2[pixel]

    average = sum(character) / len(character)
    character = character.reshape(output_pic_dim[1], output_pic_dim[0])
    print(character)
    # performing threshold, intensities higher than average equals to 1 and the rest is zero
    _, character = cv.threshold(character, average, 1, cv.THRESH_BINARY)
    plt.imshow(character, interpolation='nearest')
    plt.show()


'''Optical Character Recognition
input desired output resolution in a tuple, default is (60,60)'''

# ocr(data[5], (7, 9))

''' the result isn't satisfying and as anticipated
in this part each pixel is trained on its own based on input vector
therefore, the output picture isn't necessarily perfect as it should be
some of the pixel intensities fall below the thresh they get deleted 
in the previous part each character is decided based on the collective 
pixel intensities and some of the inaccuracies and flaws were compromised 
this way. besides, the threshold in which the pixels should be 
compared to is hard to find here. If we set it to zero results are mostly
a white screen with one or two black pixels. '''

'''#### Part 2 Section B
applying some noise'''
#
# img2 = cv.imread('resized/data18.png')
# img2 = cv.erode(img2, (1, 1), iterations=1)
# # img2 = cv.dilate(img2, (1, 1), iterations=1)
# cv.imwrite('test2b.png', img2)
# img2 = imgtodata(img2, 'binary')
# ocr(img2, (7, 9))

'''it outputs characters eventhough they are accompanied with noise
the more noise we apply the harder it gets for us to read the output 
letter, some of the output letters are still readable like C and D
and some of them not, like K,A and B'''

'''#### Part 2 Section C'''

# img2 = cv.imread('new/new (8).png')
# img2 = cv.resize(img2, (11, 15))
# img2 = cv.erode(img2, (1, 1), iterations=1)
# _,img2 = cv.threshold(img2, 120, 255, cv.THRESH_BINARY)
# cv.imwrite('test2c.png', img2)
# img2 = imgtodata(img2, 'binary')
# ocr(img2, (7, 9))

''' the network tries to output the character as of those it was trained for
it tires to find the closest possible letter to the input.
as it doesnt work as a whole nit and each neuron acts on its own,the output
is sometimes uncomprehendable. 
for example, in case of letter L, most of the neurons think the input vector 
is close to E except some handful of pixels. the result is somewhat close to E
the neurons try to see which letter is closest to the input vector and then 
output picture is closer to that trained letter.
as expected the network is not able to recognize letters other than it was 
trained with'''
