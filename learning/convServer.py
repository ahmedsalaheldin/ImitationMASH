"""This implementation is heavily based on the convolutional neural network implementation in deep learning tutorials using Theano.
https://github.com/lisa-lab/DeepLearningTutorials.git
"""
import os
import sys
import time
import zmq
import pickle
import json

import numpy
import re, numpy as np
from itertools import islice
import csv
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano import shared

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
import pylab
from scipy.stats import entropy


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def relu(x):
    return theano.tensor.switch(x<0, 0, x)

#################################################################################
#################################################################################


nkerns=[20, 50, 70]
batch_size=1

nname='params.pickle' # file containing parameters of trained network

rng = numpy.random.RandomState(23455)

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

# start-snippet-1
x = T.vector('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                # [int] labels

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'


layer0_input = x.reshape((batch_size, 1, 90, 120))

layer0 = LeNetConvPoolLayer(
rng,
input=layer0_input,
image_shape=(batch_size, 1, 90, 120),
filter_shape=(nkerns[0], 1, 7, 9),
poolsize=(2, 2)
)

layer1 = LeNetConvPoolLayer(
rng,
input=layer0.output,
image_shape=(batch_size, nkerns[0], 42, 56),
filter_shape=(nkerns[1], nkerns[0], 5, 5),
poolsize=(2, 2)
)


layer15 = LeNetConvPoolLayer(
rng,
input=layer1.output,
image_shape=(batch_size, nkerns[1], 19, 26),
filter_shape=(nkerns[2], nkerns[1], 4, 5),
poolsize=(2, 2)
)

layer2_input = layer15.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
rng,
input=layer2_input,
n_in=nkerns[2] * 8 * 11,
n_out=500,
activation=relu
#activation=T.tanh
)

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=3)


pred_model = theano.function(
[x],
layer3.y_pred,
on_unused_input='ignore'
)

get_conf = theano.function(
[x],
layer3.p_y_given_x,
on_unused_input='ignore'
)


# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer15.params + layer1.params + layer0.params



########set params from pickle file ##########
# Load a trained network to use for predictions
with open(nname, 'r') as f:
	paramsP = pickle.load(f)

dummyinput=T.iscalar('dummyinput')

loadparams = [
(p_i, pp_i)
for p_i, pp_i in zip(params, paramsP)
]

updateparams = theano.function([dummyinput], params, updates=loadparams,on_unused_input='ignore')


updateparams(1)
#############################################


def predict(instance):

	pred = pred_model(instance)

	###if using active learning###
	if len(sys.argv>1):
		conf = get_conf(instance)[0]
		ent= entropy(conf)

		if sys.argv[1]=="-a" and ent>0.9:
			return "QUERY"

	if pred[0]==0:
		return "TURN_LEFT"
	if pred[0]==1:
		return "TURN_RIGHT"
	if pred[0]==2:
		return "GO_FORWARD"

     
	
# connect to agent
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


while True:

	#receive frame from agent
	message = socket.recv()
	dec = json.loads(message)
	instance =np.float32(np.asarray(dec["A"]))
	reply=predict(instance) # get prediction
	
	print reply

	socket.send(reply)#send prediction to agent

