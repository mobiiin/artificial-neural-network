import numpy as np
import cv2 as cv
import operator
from matplotlib import pyplot as plt
#
# y1 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # neuron 1
# y2 = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # neuron 2
# y3 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# y4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# y5 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
# y6 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
# y7 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
# y_out1 = np.array([y1, y2, y3, y4, y5, y6, y7])
# # output indexing: [neuron][image]
# y_out1.reshape(7, 21)

# plt.imshow(y_out1, interpolation='nearest')
# plt.show()
# weights = np.zeros((21, 7))
# r1 = np.array([1,2,3])
# r2 = np.array([4,4,4])
# print(r1*r2)
# print(np.dot(r1,r2))

# a = np.array([-5, -20, 1])
# indices = [i for i,v in enumerate(a >= 0) if v]
# print(np.squeeze(indices))

a=[]
if bool(a) is False:
    print(a)


