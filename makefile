CC=g++

CPPFLAGS  = -O3 -Wno-write-strings -std=c++03
CPPFLAGS += -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib
CPPFLAGS += -I/usr/local/cuda-11.8/include
LDFLAGS   = -L/usr/lib/openmpi/lib -lmpi -lmpi_cxx -lm -lpthread
LDFLAGS  += -L/usr/local/cuda-11.8/lib64
LDFLAGS  += -lcudart
CPPFLAGS += -DCROSS_GPU
LDFLAGS  += -lcurl

all : main

main : main.o mpi_json_info.o pugixml.o json_to_xml.o json.o curlCall.o faunadb.o
	$(CC) -o $@ $^ $(LDFLAGS)

json.hpp : glue.hpp
json.cpp : json.hpp
main.cpp : mpi_json_info.h glue.hpp
mpi_json_info.cpp : mpi_json_info.h json.hpp

%.o : %.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<
