This repository presents a framework to train intelligent agents to perform navigation tasks using deep learning.
The agent is trained to perform navigation tasks from demonstrations.
The agent can then perform these tasks autounoumously in a 3D simulator.

The agents act within a 3D simulator, *mash-simulator* http://github.com/idiap/mash-simulator

##usage

This repository consists of a number of modules.

-The simulator acts as a server that builds an environment for a selected task. Client agents can then act in that environment by sending an action to the server. The simulator calculates changes to the world based on the received action and sends the current status (in the form of an image) back to the client.

-A stand alone deep learning module learns a mapping between observation( an image) and action from a collected dataset. The learned model is saved and can be used at any time by an agent. 

-The prediction server receives an image from the client and returns a predicted action based on a trained model.

-The agent Client which interacts with the simulation and prediction servers.

A typical workflow of the system looks like this:
	-Data collection
		-A client connects to the simulation server
		-The client receives frames from the server along with the optimal actions to take
		-The client saves the observation and action pair to the training dataset
		-After a sufficient amount of training samples is collected this process is terminated
	-Training a model
		-The training script uses the saved dataset to train a neural network.
		-The trained model is saved.
	-Performing the task
		-A client connects to the simulation server
		-The client also connects to a prediction server which loads the saved neural network.
		-The client receives a frame from the simulator and sends it to the prediction server.
		-The client receives an action from the prediction server and sends it to the simulator
		-This process is repeated until the task is completed or as long as desired.

Active learning can be enabled at the prediction server. In that case, when the prediction server is not confident about a prediction, it will send a query rather than an action to the client. The client then uses the optimal actions provided by the simulator and saves the frame + action pair to a dataset.

## Implementation details

ImitationCNN.py Contains script for training neural network from a dataset

convServer.py The prediction server that loads a trained model.

Client.cpp The client used to interact with the simulation and prediction servers


## Dependencies

A number of dependicies are needed to run this project. Such as the 3d simulator, the deep learning library required to run the training script and IPC libraries to communicate infromation between different processes.
Required repositories are downloaded and built automatically as submodules, however libraries that require system installation must be manually setup. 

Following is a list of the required dependencies:

ZeroMQ: A library for interprocess communication. Used to to send and receive messages between the agent and the prediction server. Python and C++ bindings are needed
http://zeromq.org/intro:get-the-software

Ogre: The graphics engine behind mash-simulator has some dependecies of its own
http://www.ogre3d.org/tikiwiki/Prerequisites?tikiversion=Linux

Theano: The deep learning library used to train the agent
http://deeplearning.net/software/theano/install.html


rapidjson: A library for serializing messages to be send between processes
https://pypi.python.org/pypi/python-rapidjson/

Submodules:

Mash-Simulator: The 3D simulator used to conduct the navigation experiments.
http://github.com/idiap/mash-simulator

Theano based neural network scripts.
http://github.com/lisa-lab/DeepLearningTutorials

ZeroMQ C++ headerfile
https://github.com/zeromq/cppzmq/blob/master/zmq.hpp

## Building

run scrip.sh to download the dependecies, build them and build the agent client
ImitationMASH$ ./script

## Running
run runServer.sh to start the simulator server
ImitationMASH $./runServer

To start the prediction server
ImitationMASH $ cd DeepLearningTutorials/code
code$ python convServer.py

To start the prediction server with active learning enabled
code$ python convServer.py -a

To run the training script
ImitationMASH $ cd DeepLearningTutorials/code
code$ python ImitationCNN.py

To run the Client to collect data (and act using the optimal actions provided by the server)
ImitationMASH $ cd build
build$ ./client

To run the client with prediction enabled (uses predictions from the prediction server to act)
build$ ./client -p
