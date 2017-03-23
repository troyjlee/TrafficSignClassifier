import pickle
import numpy as np
import tensorflow as tf
import cv2
import math
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

################################
# Preprocessing functions
################################
def jitter(X, angle_limit = 10, trans_limit = 2, scale_limit = 0.1, bright_limit = 0.2):
    '''
    This function creates an "infinite" data set by modifying the 
    training images.  We rotate, translate, rescale, and 
    scale the brightness by random amounts.
    '''
    shape = X.shape
    Y = np.empty(shape)
    rows, cols = shape[1], shape[2]
    center = np.array([[cols/2.], [rows/2.]])
    for i in range(shape[0]):
        img = X[i]
        angle = np.random.uniform(low = -1, high = 1)*angle_limit*math.pi/180
        trans = np.random.uniform(low = -1, high = 1, size = (2,1))*trans_limit
        scale = np.random.uniform(low = -1, high = 1)*scale_limit + 1
        alpha = scale*math.cos(angle)
        beta = scale*math.sin(angle)
        M = np.array([[alpha, -beta],[beta, alpha]], np.float32)
        # warpAffine expects a 2-by-3 matrix [M|b], and performs Mx + b
        # We want M(x-center) + center + trans, thus b = center + trans - M*center
        b = center + trans - np.matmul(M,center)
        W = np.concatenate((M,b),1)
        new_img = cv2.warpAffine(img,W,(cols,rows),borderMode=cv2.BORDER_REFLECT_101)
        yuv = cv2.cvtColor(new_img, cv2.COLOR_RGB2YUV)
        y,u,v = cv2.split(yuv)
        y = y*(1 + bright_limit*np.random.uniform(low = -1, high = 1))
        y = np.clip(y,0,255)
        yuv = cv2.merge([y,u,v])
        new_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        Y[i] = new_img
    return(Y)



#################################
# general model functions
#################################
def xavier(shape):
  '''
  This function returns the standard deviation needed for the 
  initialization of He et al. good for ReLus
  '''
  shape = np.array(shape)
  prod = np.prod(shape[:-1])
  return np.sqrt(2/prod)

def dense(input, size):
    dims = [input.get_shape()[1], size]
    weights = tf.get_variable('weights', shape = dims, 
        initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape = [size], 
        initializer = tf.constant_initializer(0.1))
    return tf.matmul(input, weights) + biases

def dense_relu(input, size):
    return tf.nn.relu(dense(input, size))

def conv_relu(input, depth, kernel_size = 3, pad = 'SAME', stride = 1):
    dims = [kernel_size, kernel_size, input.get_shape()[3], depth]
    weights = tf.get_variable('weights', shape = dims, 
        initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape = [depth],     
        initializer = tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(input, weights, strides = [1,stride,stride,1], padding = pad, name = 'conv')
    return tf.nn.relu(conv + biases)

def pool(input, size = 2):
    return tf.nn.max_pool(input, ksize = [1, size, size, 1], 
        strides = [1, size, size, 1], padding = 'SAME')
#################################################

def net(x):
    '''
    definition of our model.  We first use a 1x1 convolution to 
    the 3 channel input into one channel.

    Then there are 3 blocks of 2 consecutive 3x3 convolutions.

    The output of each block is flattened, concatenated and fed into 
    a dense layer.
    '''
    with tf.variable_scope('conv1a'):
        conv1a = conv_relu(x, 16) 
    with tf.variable_scope('conv1b'):
        conv1b = conv_relu(conv1a, 16) 
        pool1 = pool(conv1b)

    with tf.variable_scope('conv2a'):
        conv2a = conv_relu(pool1, 32)
    with tf.variable_scope('conv2b'):
        conv2b = conv_relu(conv2a, 32)
        pool2 = pool(conv2b)

    with tf.variable_scope('conv3a'):
        conv3a = conv_relu(pool2, 64)
    with tf.variable_scope('conv3b'):
        conv3b = conv_relu(conv3a, 64)
        pool3 = pool(conv3b)

    # prepare outputs from each conv layer to be fed into dense layers

    # 1st layer output
    pool1 = pool(pool1, size = 4)
    pool1 = flatten(pool1)

    # 2nd layer output
    pool2 = pool(pool2)
    pool2 = flatten(pool2)

    # 3rd layer output
    pool3 = flatten(pool3)

    v = tf.concat(1, [pool1, pool2, pool3])
    v_drop = tf.nn.dropout(v, keep_prob)

    with tf.variable_scope('dense4'):
        dense4 = dense_relu(v_drop, 512)
        dense4_drop = tf.nn.dropout(dense4, keep_prob)

    with tf.variable_scope('logits'):
        logits = dense(dense4_drop, 43)
   
    return logits

##############################################
training_file = './traffic-signs-data/extended_train.p'
validation_file= './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

X_train = X_train/255 - 0.5
X_valid = X_valid/255 - 0.5
X_test = X_test/255 - 0.5


# parameters
EPOCHS = 10
BATCH_SIZE = 128
# initial rate 0.001
rate = 0.001

# define placeholders
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

logits = net(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, prob):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        iter = int(offset/BATCH_SIZE)
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy,loss = sess.run([accuracy_operation, loss_operation],
            feed_dict={x: batch_x, y: batch_y, keep_prob:prob})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    total_accuracy = total_accuracy / num_examples
    total_loss = total_loss / num_examples
    return total_accuracy, total_loss

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, './vgg_aug')
    num_examples = len(X_train)
    print(num_examples)
    num_batches = int(np.ceil(num_examples/BATCH_SIZE))

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        X_jitter = jitter(X_train)
        total_acc = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            iter = int(offset/BATCH_SIZE)
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_jitter[offset:end], y_train[offset:end]
            _, acc = sess.run([training_operation, accuracy_operation],
                feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            total_acc += (acc * len(batch_x))
        validation_accuracy, validation_loss = evaluate(X_valid, y_valid, 1.0)
        total_acc = total_acc/num_examples

        # https://stackoverflow.com/questions/37902705/how-to-manually-create-a-tf-summary
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training Accuracy = {:.3f}".format(total_acc))
        print()

    saver.save(sess, './vgg3')
    print("Model saved")
