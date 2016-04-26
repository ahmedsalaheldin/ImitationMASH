"""This implementation is heavily based on the convolutional neural network implementation in deep learning tutorials using Theano.
https://github.com/lisa-lab/DeepLearningTutorials.git
"""
import os
import sys
import time
import pickle

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




def get_batch(reader, index, batch_size, datatype):

   
    if isinstance( index, int ) == False :
	index =0
	print "invalid index set to 0"

    arr = []
    start = index * batch_size
    stop = (index + 1) * batch_size

    for row in islice(reader,start,stop):
	   arr.append(row)

    x=np.asarray(arr, dtype=datatype)
    if datatype == np.int32 :
	return np.ravel(x)

    return x

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

def evaluate_lenet5(learning_rate=0.001, n_epochs=500,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50, 70], batch_size=300):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """



    rng = numpy.random.RandomState(23455)


    dname = 'dataset.csv'   # file containing frames
    tname = 'target.csv'    # file contatining actions
    nname = 'params.pickle' # file to write the trained network parameters

    n_train_batches=20794 / batch_size
    n_valid_batches=20794 / batch_size
    n_test_batches =40000 / batch_size






    print "test model batches" , batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
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

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

	
    # create a function to compute the mistakes that are made by the model

    test_model = theano.function(
        [x,y],
        layer3.errors(y),
	on_unused_input='ignore'
    )

    validate_model = theano.function(
        [x,y],
        layer3.errors(y),
	on_unused_input='ignore'
    )


    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer15.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x,y],
        cost,
        updates=updates,
	on_unused_input='ignore'
    )



    
    ########set params from pickle file ##########
    #If initializing the weights using previously trained weights
    '''with open('params.pickle', 'r') as f:
	paramsP = pickle.load(f)
    


    dummyinput=T.iscalar('dummyinput')

    loadparams = [
        (p_i, pp_i)
        for p_i, pp_i in zip(params, paramsP)
    ]

    updateparams = theano.function([dummyinput], params, updates=loadparams,on_unused_input='ignore')


    updateparams(1)
	'''
    #############################################

    dfv= open('datasetL2ConfAug.csv', 'rt')
    tfv= open('targetL2ConfAug.csv', 'rt')  
    datasetv = csv.reader(dfv, delimiter=',')
    targetv = csv.reader(tfv, delimiter=',')

    validation_losses = []
    for i in xrange(n_valid_batches):
	testbatchx = get_batch(datasetv, 0, batch_size, np.float32)
	testbatchy = get_batch(targetv, 0, batch_size, np.int32)
	validation_losses.append(validate_model(testbatchx,testbatchy))
	

    this_validation_loss = numpy.mean(validation_losses)
    print "random valid score = " , this_validation_loss*100.
    print "random networks score = " , validation_losses[1:10]



    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 1000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

	dftr= open('datasetL2ConfAug.csv', 'rt')
	tftr= open('targetL2ConfAug.csv', 'rt')  
	datasettr = csv.reader(dftr, delimiter=',')
	targettr = csv.reader(tftr, delimiter=',')


        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
               print 'training @ iter = ', iter

	    batchx = get_batch(datasettr, 0, batch_size, np.float32)
	    batchy = get_batch(targettr, 0, batch_size, np.int32)

            cost_ij = train_model(batchx,batchy)
	    

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
		dfv= open('datasetL2ConfAug.csv', 'rt')
		tfv= open('targetL2ConfAug.csv', 'rt')  
		datasetv = csv.reader(dfv, delimiter=',')
		targetv = csv.reader(tfv, delimiter=',')
		validation_losses = []
		for i in xrange(n_valid_batches):
			batchx = get_batch(datasetv, 0, batch_size, np.float32)
			batchy = get_batch(targetv, 0, batch_size, np.int32)
			validation_losses.append(validate_model(batchx,batchy))

		this_validation_loss = numpy.mean(validation_losses)


                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
					
		    #save the trained weights
		    with open('params.pickle', 'w') as f:
    			 pickle.dump(params, f)

                    # test it on the test set
		    print "test model index" , i
		    print "test model batches" , batch_size

		    if testdata!=null:
			    dft= open(testdata, 'rt')
			    tft= open(testtarget, 'rt')  
			    datasett = csv.reader(dft, delimiter=',')
			    targett = csv.reader(tft, delimiter=',')
			    test_losses = []
	    		    for i in xrange(n_test_batches):
				testbatchx = get_batch(datasett, 0, batch_size, np.float32)
				testbatchy = get_batch(targett, 0, batch_size, np.int32)
				test_losses.append(test_model(testbatchx,testbatchy))
	   
	  		    test_score = numpy.mean(test_losses)

		            print(('     epoch %i, minibatch %i/%i, test error of '
		                   'best model %f %%') %
		                  (epoch, minibatch_index + 1, n_train_batches,
		                   test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
