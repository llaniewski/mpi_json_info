#ifndef MPI_JSON_INFO_H
#define MPI_JSON_INFO_H

#include <mpi.h>
#include <string>

std::string nodeName(MPI_Comm comm);
std::string cpuJSON();
std::string nodeJSON(MPI_Comm comm);
std::pair< std::string, int > procJSON(int node);
std::string nodesJSON(MPI_Comm comm, bool detailed);
std::string reformatJSON(const std::string& info);
std::string stripJSON(const std::string& info);
std::string versionJSON(const int& major, const int& minor);
std::string compilationJSON();
std::string runtimeJSON();

#endif
