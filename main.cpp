#include <stdio.h>
#include <mpi.h>
#include <vector>
#include <fstream>

std::string nodeName(MPI_Comm comm) {
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	int cpname_len;
	char cpname[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(cpname, &cpname_len);
	if (rank > 2) {
		cpname[cpname_len] = 'T';
		cpname[cpname_len+1] = 'M';
		cpname[cpname_len+2] = '\0';
		cpname_len += 2;
	} else {
		cpname[cpname_len] = '\0';
	}
	return std::string(cpname, cpname_len);
}

std::string cpuJSON() {
	std::string ret;
	std::string cores;
	std::string cpus;
	std::ifstream f("/proc/cpuinfo");
	if (f.fail()) {
		ret = ret + "\"error\": \"/proc/cpuinfo: " + strerror(errno) + "\"";
	} else {
		bool first = true;
		std::string corename = "unknown";
		std::string core;
		std::string cpuname = "unknown";
		std::string cpu;
		std::string line;
		while (std::getline(f, line)) {
			if (line == "") {
				core = "\"" + corename + "\": { " + core + " }";
				if (!first) ret = ret + ", ";
				first = false;
				cores = cores + proc;
				proc = "";
			} else {
				size_t i,j;
				i = line.find(":");
				if (i == std::string::npos) continue;
				for(j = i; j > 0; j--) if (line[j-1] != ' ' && line[j-1] != '\t') break;
				if (i < line.size()-1) i++;
				std::string tag = line.substr(0,j);
				std::string val = line.substr(i+1,line.size()-i-1);
				if (tag == "processor") {
					procnum = val;
					proc = proc + "\"number\": " + val;
				} else if (tag == "model name") {
					proc = proc + ", \"name\": \"" + val + "\"";
				} else if (tag == "physical id") {
					proc = proc + ", \"physical\": " + val;
				} else if (tag == "cache size") {
					proc = proc + ", \"cache\": \"" + val + "\"";
				} else if (tag == "ventor id") {
					proc = proc + ", \"ventor\": \"" + val + "\"";
				} else if (tag == "siblings") {
					proc = proc + ", \"siblings\": \"" + val + "\"";
				} else if (tag == "cpu MHz") {
					proc = proc + ", \"freq\": " + val;
				}
			}
		}
		f.close();
	}
	ret = "{ " + ret + " }";
	return ret;
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
	std::string ret = "{ ";
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	std::string pname = nodeName(comm);
	int wrank = rank;
	int firstrank = 0;
	while (true) {
		std::string otherpname = MPI_Bcast(pname, firstrank, comm);
		if (otherpname == pname) {
			wrank = size;
		}
		std::string nodejson;
		if (rank == firstrank) nodejson = nodeJSON(comm);
		nodejson = MPI_Bcast(nodejson, firstrank, comm);
		ret = ret + "\"" + otherpname + "\": " + nodejson;
		MPI_Allreduce(&wrank, &firstrank, 1, MPI_INT, MPI_MIN, comm );
		if (firstrank >= size) break;
		ret = ret + ", ";
	}
	ret = ret + " }";
	return ret;
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