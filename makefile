CXX=CC

CXXFLAGS  = -O3 -Wno-write-strings -std=c++11 -fopenmp --rocm-path=${ROCM_PATH} -x hip
#CXXFLAGS += -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib
#CXXFLAGS += -I/usr/local/cuda-11.8/include
#LDFLAGS   = -L/usr/lib/openmpi/lib -lmpi -lmpi_cxx
LDFLAGS  += -lm -lpthread
#LDFLAGS  += -L/usr/local/cuda-11.8/lib64
#LDFLAGS  += -lcudart -lnvidia-ml
#CXXFLAGS += -DCROSS_GPU
CXXFLAGS += -DCROSS_HIP #--offload-arch=gfx90a
LDFLAGS  += -lcurl

all : main

main : main.o mpi_json_info.o pugixml.o json_to_xml.o json.o curlCall.o faunadb.o
	$(CXX) -o $@ $^ $(LDFLAGS)

json.hpp : glue.hpp
json.o : json.cpp json.hpp
main.cpp : mpi_json_info.h glue.hpp
mpi_json_info.o : mpi_json_info.cpp mpi_json_info.h json.hpp glue.hpp

%.o : %.cpp makefile
	$(CXX) $(CXXFLAGS) -c -o $@ $<
