#ifndef MPI_JSON_INFO_H
#define MPI_JSON_INFO_H

#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <string>
#include "json.hpp"

std::string nodeName(MPI_Comm comm);
JSON cpuJSON();
JSON nodeJSON(MPI_Comm comm);
std::pair< JSON, int > procJSON(int node);
JSON nodesJSON(MPI_Comm comm, bool detailed);
JSON versionJSON(const int& major, const int& minor);
JSON compilationJSON();
JSON runtimeJSON();
JSON localJSON();

#endif
