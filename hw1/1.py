import numpy as np
import glob
import cv2 as cv
import os
import operator


def imgtodata(img, method):
    img_list = []
    for pixel in img:
        for r, g, b in pixel:
            if r > 128:
                img_list.append(-1) if method == 'bipolar' else img_list.append(0)
            else:
                img_list.append(1)
    return img_list


# creating the dataset and output value
def dataset(width, hight, method='bipolar', save=False):
    y1 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # neuron 1
    y2 = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # neuron 2
    y3 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y5 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y6 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    y7 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_out1 = np.array([y1, y2, y3, y4, y5, y6, y7])
    y_out1.reshape(7, 21)
    if method == 'bipolar':
        y_out1 = np.where(y_out1 == 0, -1, 1)

    data1 = []
    dir_path = os.getcwd() + '\\data\\'
    i = 0
    kernel = np.ones((5, 5), np.uint8)

    for img in glob.glob(dir_path + "/*.png"):
        img = cv.imread(img)
        img_dilation = cv.erode(img, kernel, iterations=1)
        img = cv.resize(img_dilation, (width, hight))
        _, thresh1 = cv.threshold(img, 120, 255, cv.THRESH_BINARY)
        if save:
            cv.imwrite("resized/data%i.png" % i, thresh1)

        data1.append(imgtodata(thresh1, method))
        i += 1

    data1 = np.array(data1)
    data1.reshape(21, width * hight)

    return data1, y_out1


def train(inputs, output):
    # weight indexing: [neuron][pixels]
    weight = np.zeros((7, len(inputs[0])), dtype=int)
    bias1 = np.zeros(7, dtype=int)
    for neuron in range(7):
        for image in range(21):
            weight[neuron] += inputs[image] * output[neuron][image]
            bias1[neuron] += output[neuron][image]
    return weight, bias1


def recognize(img, weight, bias1):
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
        elif neurons == 6 and found == False:
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


'''# data[0] is pic num 1 # data.shape = 21*63'''
# data, y_out  = dataset(7, 9, 'binary')

'''##### Part2 Section B
# training each neuron with one character'''
# weights, bias = train(data, y_out)
# checking the trained weights
# recognize(data[5], weights, bias)
''' these neurons cannot recognize the input character correctly
cause we used binary method to train the weights 
in each case the y_in for all of the neurons yields
more than thresh which is not acceptable    '''

''' applying some noise to input image'''
# img1 = cv.imread('resized/data7.png')
# img1 = cv.erode(img1, (1, 1), iterations=1)
# img1 = imgtodata(img1, 'binary')
# recognize(img1, weights, bias)
''' this part still doesnt work same as previous part
the reason as I mentioned earlier is using binary method
in training the weights'''

'''##### Part1 Section C
# training all neurons with bipolar data'''
# data, y_out = dataset(7, 9, 'bipolar')
# weights, bias = train(data, y_out)
# recognize(data[20], weights, bias)
''' as we see here the networks is more cabpable of recognizing
which character is being fed into, although its not fully functional
it realizes the A C K J characters but still has a hard time 
figuring out couple of letters like B and E'''
# applying some noise to input image
# img1 = cv.imread('resized/data18.png')
# img1 = cv.erode(img1, (1, 1), iterations=1)
# img1 = imgtodata(img1, 'binary')
# recognize(img1, weights, bias)
''' applying noise to some of the input images that the network 
was able to recognize earlier, we see that some of the images 
are still recognized with noise like image num 18 and some of 
them are nolonger recognizable'''

##### Part 2
data, y_out = dataset(11, 15, 'bipolar')
weights, bias = train(data, y_out)
recognize(data[18], weights, bias)

# def ocr()

weight2 = np.zeros((60*60, 7), dtype=int)
bias2 = np.zeros(60*60, dtype=int)
y_out2, _ = dataset(60, 60, 'binary', True)
