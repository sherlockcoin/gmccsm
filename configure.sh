// extracflags="-march=native -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"
// CFLAGS="-O3 -Xcompiler -Wall" ./configure CXXFLAGS="-O3 $extracflags" --with-cuda=/usr/local/cuda --with-nvml=libnvidia-ml.so
// slower with the compile options above - me

./configure "CFLAGS=-O3" "CXXFLAGS=-O3"
### --with-cuda=/usr/local/cuda-8.0
