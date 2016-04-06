#intilize and update submodules
git submodule init
git submodule update

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
