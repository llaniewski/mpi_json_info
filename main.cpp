#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <vector>
#include <fstream>
#include <cerrno>
#include "glue.hpp"

std::string nodeName(MPI_Comm comm) {
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	int cpname_len;
	char cpname[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(cpname, &cpname_len);
/*	if (rank > 2) {
		cpname[cpname_len] = 'T';
		cpname[cpname_len+1] = 'M';
		cpname[cpname_len+2] = '\0';
		cpname_len += 2;
	} else { */
		cpname[cpname_len] = '\0';
/*	} */
	return std::string(cpname, cpname_len);
}

std::string cpuJSON() {
	Glue ret(", ","{ ", " }");
	Glue physicalid(", ","[ ", " ]");
	Glue coreid(", ","[ ", " ]");
	Glue cpus(", ","[ ", " ]");
	std::ifstream f("/proc/cpuinfo");
	if (f.fail()) {
		ret << "\"error\": \"/proc/cpuinfo: " << strerror(errno) << "\"";
	} else {
		std::string cpuname = "-1";
		std::string corename = "-1";
		int cpunumber = -1;
		Glue cpu(", ","{ ", " }");
		std::string line;
		while (std::getline(f, line)) {
			if (line == "") {
				physicalid << cpuname;
				coreid << corename;
				int k = atoi(cpuname.c_str());
				if (k > cpunumber) {
					cpus << cpu.str();
					cpunumber = k;
				}
				cpu.clear();
			} else {
				size_t i,j;
				i = line.find(":");
				if (i == std::string::npos) continue;
				for(j = i; j > 0; j--) if (line[j-1] != ' ' && line[j-1] != '\t') break;
				if (i < line.size()-1) i++;
				std::string tag = line.substr(0,j);
				std::string val = line.substr(i+1,line.size()-i-1);
				if (tag == "model name") {
					cpu << "\"name\": \"" + val + "\"";
				} else if (tag == "physical id") {
					cpuname = val;
				} else if (tag == "cache size") {
					cpu << "\"cache\": \"" + val + "\"";
				} else if (tag == "cpu cores") {
					cpu << "\"cores\": \"" + val + "\"";
				} else if (tag == "core id") {
					corename = val;
				} else if (tag == "ventor id") {
					cpu << "\"ventor\": \"" + val + "\"";
				} else if (tag == "siblings") {
					cpu << "\"vcores\": \"" + val + "\"";
				}
			}
		}
		f.close();
	}
	ret << "\"vcore_to_cpu\": " + physicalid.str();
	ret << "\"vcore_to_core\": " + coreid.str();
	ret << "\"cpus\": " + cpus.str();
	return ret.str();
}

std::string nodeJSON(MPI_Comm comm) {
	return "{ \"name\": \"" + nodeName(comm) + "\", \"cpu\": " + cpuJSON() + " }";
}

std::string MPI_Bcast(const std::string& str, int root, MPI_Comm comm) {
	size_t size = str.size();
	MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, comm);
	char * buf = new char[size+1];
	strcpy(buf, str.c_str());
	MPI_Bcast(buf, size+1, MPI_CHAR, root, comm);
	std::string ret(buf,size);
	delete[] buf;
	return ret;
}

std::string nodesJSON(MPI_Comm comm) {
	Glue nodes(", ", "[ ", " ]");
	Glue ranks(", ", "[ ", " ]");
	Glue ret(", ", "{ ", " }");
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	std::string pname = nodeName(comm);
	int wrank = rank;
	int firstrank = 0;
	int mynode = -1;
	int i = 0;
	while (true) {
		std::string otherpname = MPI_Bcast(pname, firstrank, comm);
		if (otherpname == pname) {
			wrank = size;
			mynode = i;
		}
		std::string nodejson;
		if (rank == firstrank) nodejson = nodeJSON(comm);
		nodejson = MPI_Bcast(nodejson, firstrank, comm);
		nodes << nodejson;
		i++;
		MPI_Allreduce(&wrank, &firstrank, 1, MPI_INT, MPI_MIN, comm );
		if (firstrank >= size) break;
	}
	ret << "\"ranks\": " << ranks.str();
	ret << "\"nodes\": " << nodes.str();
	return ret.str();
}

int main ( int argc, char * argv[] )
{
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	std::string info = nodesJSON(comm);
	if (rank == 0) printf("%s\n", info.c_str());
	MPI_Finalize();
}