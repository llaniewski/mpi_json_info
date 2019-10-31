#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <vector>
#include <fstream>
#include <cerrno>
#include "glue.hpp"

std::string nodeName(MPI_Comm comm) {
	int cpname_len;
	char cpname[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(cpname, &cpname_len);
	return std::string(cpname, cpname_len);
}

std::string cpuJSON() {
	Glue ret(", ","{ ", " }");
	Glue physicalid(", ","[ ", " ]");
	Glue coreid(", ","[ ", " ]");
	Glue cpus(", ","[ ", " ]");
	std::ifstream f("/proc/cpuinfo");
	if (f.fail()) {
		std::string err = strerror(errno);
		ret << "\"error\": \"/proc/cpuinfo: " + err + "\"";
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

std::pair< std::string, int > procJSON(int node) {
	int core = -1;
	Glue ret(", ", "{ ", " }");
	ret << (Glue() << "\"node\": " << node).str();
	std::ifstream f("/proc/self/stat");
	if (f.fail()) {
		std::string err = strerror(errno);
		ret << "\"error\": \"/proc/self/stat: " + err + "\"";
	} else {
		std::string line;
		int i = 0;
		while (std::getline(f, line, ' ')) {
			i++;
			if (i == 39) {
				core = atoi(line.c_str());
				ret << "\"vcore\": " + line;
			} else if (i == 1) {
				ret << "\"pid\": " + line;
			} else if (i == 14) {
				ret << "\"utime\": \"" + line + "\"";
			} else if (i == 15) {
				ret << "\"stime\": \"" + line + "\"";
			} else if (i == 22) {
				ret << "\"starttime\": \"" + line + "\"";
			} else if (i == 23) {
				ret << "\"vsize\": \"" + line + "\"";
			}
		}
		f.close();
	}
	return std::make_pair( ret.str(), core );
}

std::string nodesJSON(MPI_Comm comm, bool detailed) {
	Glue nodes(", ", "[ ", " ]");
	Glue ret(", ", "{ ", " }");
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	ret << (Glue() << "\"size\": " << size).str();
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
	std::pair< std::string, int > procjson = procJSON(mynode);
	int core = procjson.second;
	if (detailed) {
		Glue ranks(", ", "[ ", " ]");
		for (int i=0; i<size; i++) {
			std::string otherproc;
			otherproc = MPI_Bcast(procjson.first, i, comm);
			ranks << otherproc;
		}
		ret << "\"ranks\": " + ranks.str();
	}
	
	int* perrank = NULL;
	if (rank == 0) perrank = new int[size];
	MPI_Gather(&core, 1, MPI_INT, perrank, 1, MPI_INT, 0, comm);
	if (rank == 0) ret << "\"rank_to_vcore\": " + (Glue(", ", "[ ", " ]") << std::make_pair(perrank,size)).str();
	MPI_Gather(&mynode, 1, MPI_INT, perrank, 1, MPI_INT, 0, comm);
	if (rank == 0) ret << "\"rank_to_node\": " + (Glue(", ", "[ ", " ]") << std::make_pair(perrank,size)).str();
	if (rank == 0) delete[] perrank;
	ret << "\"nodes\": " + nodes.str();
	return ret.str();
}

std::string reformatJSON(const std::string& info) {
	std::string info_formated;
	int ind = 0;
	bool in_quote = false;
	int nl_after = -1;
	int nl_before = -1;
	bool space_after = false;
	bool hold_nl = false;
	for (size_t i = 0; i < info.size(); i++) {
		char c = info[i];
		if (in_quote) {
			if (c == '"') in_quote = false;
		} else if (c == '"') {
			in_quote = true;
		} else if (c == ' ') {
			continue;
		} else if (c == ':') {
			space_after = true;
		} else if (c == '{') {
			if (hold_nl) {
				nl_before = ind;
				hold_nl = false;
			}
			ind++;
			nl_after = ind;
		} else if (c == '}') {
			ind--;
			nl_before = ind;
		} else if (c == '[') {
			ind++;
			hold_nl = ind;
		} else if (c == ']') {
			ind--;
			if (hold_nl) hold_nl = false; else nl_before = ind;
		} else if (c == ',') {
			if (hold_nl) space_after = true; else nl_after = ind;
		}
		if (nl_before >= 0) {
			info_formated.push_back('\n');
			for (int j = 0; j<nl_before*2; j++) info_formated.push_back(' ');
			nl_before = -1;
		}
		info_formated.push_back(c);
		if (nl_after >= 0) {
			info_formated.push_back('\n');
			for (int j = 0; j<nl_after*2; j++) info_formated.push_back(' ');
			nl_after = -1;
		}
		if (space_after) {
			info_formated.push_back(' ');
			space_after = false;
		}
	}
	return info_formated;
}

