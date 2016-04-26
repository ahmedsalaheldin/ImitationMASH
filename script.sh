#intilize and update submodules
git submodule init
git submodule update

#copy network files to relevant directories
cp Trained\ networks/params6000.pickle DeepLearningTutorials/
cp learning/{ImitationCNN.py,convServer.py} DeepLearningTutorials/code/
#replace modified Deep RL script

#intilize and update submodules for the mash-simulator
cd mash-simulator ; git submodule init ; git submodule update 

#build mash-simulator
mkdir build
cd build

cmake ..
make

cd .. ; cd ..

#build our client
mkdir build
cd build

cmake ..
make
