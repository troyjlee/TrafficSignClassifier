'''
this code enlarges the training set by adding flipped images that 
are still valid traffic signs.  we look at flipping horizontally, 
vertically, both horizontally and vertically, and taking the transpose 
(reflecting about the diagonal) and reflecting about the anti-diagonal.
The enlarged data set is pickled and saved into the given output file.
'''

import pickle
import numpy as np
import tensorflow as tf
import cv2

training_file = './traffic-signs-data/train.p'
output_file = './traffic-signs-data/extended_train.p'


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']

num_classes = 43

# images that can be flipped left to right without changing class 
flip_lr = np.array([11,12,13,15,17,18,22,26,30,35])

# images that can be vertically flipped
flip_ud = np.array([1,5,12,15,17])

#images preserved when flipped vertically and horizontally
flip_both = np.array([12,15,32,17,40])

# image pairs that when flipped left to right switch between themselves
flip_switch = np.array([[19,20],[33,34],[36,37],[38,39],[20,19],[34,33],[37,36],[39,38]])

# images preserved by transpose
flip_transpose = [12,15,32,38]

# images preserved when reflected in the anti-diagonal
flip_anti = [12,15,32,39]


X_extended = np.empty((0,32,32,3), dtype = np.float32)
y_extended = np.empty((0), dtype = np.int32)

for i in range(num_classes):
    S = X_train[y_train == i]
    num = S.shape[0]
    X_extended = np.concatenate((X_extended,S))
    y_extended = np.concatenate((y_extended,i*np.ones(num)))

    if i in flip_lr:
        Slr = S[:,:,::-1,:]
        X_extended = np.concatenate((X_extended,Slr))
        y_extended = np.concatenate((y_extended,i*np.ones(num)))

    if i in flip_ud:
        Sud = S[:,::-1,:,:]
        X_extended = np.concatenate((X_extended,Sud))
        y_extended = np.concatenate((y_extended,i*np.ones(num)))

    if i in flip_both:
        Sboth = S[:,::-1,::-1,:]
        X_extended = np.concatenate((X_extended,Sboth))
        y_extended = np.concatenate((y_extended,i*np.ones(num)))

    if i in flip_transpose:
        Str = np.transpose(S,(0,2,1,3))
        X_extended = np.concatenate((X_extended,Str))
        y_extended = np.concatenate((y_extended,i*np.ones(num)))

    if i in flip_anti:
        Sboth = S[:,::-1,::-1,:]
        Santi = np.transpose(Sboth,(0,2,1,3))
        X_extended = np.concatenate((X_extended,Santi))
        y_extended = np.concatenate((y_extended,i*np.ones(num)))

    if i in flip_switch[:,0]:
        partner = flip_switch[flip_switch[:,0] == i][0][1]
        Slr = S[:,:,::-1,:]
        X_extended = np.concatenate((X_extended,Slr))
        y_extended = np.concatenate((y_extended,partner*np.ones(num)))

extended_train = {'features':X_extended, 'labels':y_extended}
pickle.dump(extended_train, open(output_file, 'wb'))
