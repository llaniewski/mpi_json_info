CC=g++

CPPFLAGS = -I/usr/local/cuda/include -I/usr/include/mpi -O3 -Wno-write-strings
LDFLAGS = -L/usr/local/cuda/lib64 -L/usr/lib/openmpi/lib -lmpi -lm -L/net/software/local/cuda/9.0/lib64/

all : main
gpu : main
gpu : LDFLAGS += -lcudart
gpu : CPPFLAGS += -DCROSS_GPU

main : main.o mpi_json_info.o
	$(CC) -o $@ $^ $(LDFLAGS)

main.o : main.cpp mpi_json_info.h glue.hpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

mpi_json_info.o : mpi_json_info.cpp mpi_json_info.h glue.hpp
	$(CC) $(CPPFLAGS) -c -o $@ $<
