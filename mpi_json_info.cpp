#include "mpi_json_info.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <cerrno>
#include "glue.hpp"
#ifdef CROSS_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
#endif
#ifdef __GLIBC__
	#include <gnu/libc-version.h>
#endif

std::string nodeName(MPI_Comm comm) {
	int cpname_len;
	char cpname[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(cpname, &cpname_len);
	return std::string(cpname, cpname_len);
}

JSON gpuJSON() {
	JSONobject ret;
#ifdef CROSS_GPU
	cudaError_t status;
	int ngpus;
	JSONarray gpus;
	status = cudaGetDeviceCount(&ngpus);
	if (status == cudaErrorNoDevice) {
		ngpus = 0;
		status = cudaSuccess;
	}
	if (status != cudaSuccess) {
		ret << "error" << Glue::colon() << std::string("cudaGetDeviceCount: ") + std::string(cudaGetErrorString(status));
	} else {
		for (int i=0; i<ngpus; i++) {
			JSONobject gpu;
			cudaDeviceProp prop;
			status = cudaGetDeviceProperties(&prop, i);
			if (status != cudaSuccess) {
				gpu << "error" << Glue::colon() << std::string("cudaGetDeviceProperties: ") + std::string(cudaGetErrorString(status));
			} else {
				gpu << "name" << Glue::colon() << std::string(prop.name);
				gpu << "totalGlobalMem" << Glue::colon() << prop.totalGlobalMem;
				gpu << "sharedMemPerBlock" << Glue::colon() << prop.sharedMemPerBlock;
				gpu << "version" << Glue::colon() << versionJSON(prop.major,prop.minor);
				gpu << "clockRate" << Glue::colon() << prop.clockRate;
				gpu << "multiProcessorCount" << Glue::colon() << prop.multiProcessorCount;
				gpu << "ECCEnabled" << Glue::colon() << Glue::neverquote(prop.ECCEnabled ? "true" : "false");
			}
			gpus << gpu.str();
		}
		ret << "gpus:" << Glue::colon() << ngpus;
		ret << "gpu:"  << Glue::colon() << gpus.str();
	}
#else
	ret << "gpus" << Glue::colon() << 0;
	ret << "gpu" << Glue::colon() << JSONarray().str();
	ret << "warning:" << Glue::colon() << "compiled without CUDA";
#endif
	return ret.str();
}

JSON cpuJSON() {
	JSONobject ret;
	JSONarray physicalid;
	JSONarray coreid;
	JSONarray cpus;
	int ncores=0;
	int ncpu=0;
	std::ifstream f("/proc/cpuinfo");
	if (f.fail()) {
		std::string err = strerror(errno);
		ret << "error" << Glue::colon() << std::string("/proc/cpuinfo: ") + std::string(err);
	} else {
		Glue::neverquote cpuname = "-1";
		Glue::neverquote corename = "-1";
		int cpunumber = -1;
		JSONobject cpu;
		std::string line;
		while (std::getline(f, line)) {
			if (line == "") {
				ncores++;
				physicalid << cpuname;
				coreid << corename;
				int k = atoi(cpuname.c_str());
				if (k > cpunumber) {
					ncpu++;
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
					cpu << "name" << Glue::colon() << val ;
				} else if (tag == "physical id") {
					cpuname = val;
				} else if (tag == "cache size") {
					cpu << "cache" << Glue::colon() << val ;
				} else if (tag == "cpu cores") {
					cpu << "cores" << Glue::colon() << Glue::neverquote(val) ;
				} else if (tag == "core id") {
					corename = val;
				} else if (tag == "ventor id") {
					cpu << "ventor" << Glue::colon() << val ;
				} else if (tag == "siblings") {
					cpu << "vcores" << Glue::colon() << Glue::neverquote(val) ;
				}
			}
		}
		f.close();
	}
	ret << "vcores" << Glue::colon() << ncores;
	ret << "cpus" << Glue::colon() << ncpu;
	ret << "vcore_to_cpu" << Glue::colon() << physicalid.str();
	ret << "vcore_to_core" << Glue::colon() << coreid.str();
	ret << "cpu" << Glue::colon() << cpus.str();
	return ret.str();
}

JSON nodeJSON(MPI_Comm comm) {
	JSONobject ret;
	ret << "name" << Glue::colon() << nodeName(comm) ;
	ret << "cpu" << Glue::colon() <<cpuJSON();
	ret << "gpu" << Glue::colon() <<gpuJSON();
	return ret.str();
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

std::pair< JSON, int > procJSON(int node) {
	int core = -1;
	JSONobject ret;
	ret << "node" << Glue::colon() << node;
	std::ifstream f("/proc/self/stat");
	if (f.fail()) {
		std::string err = strerror(errno);
		ret << "error" << Glue::colon() << std::string("/proc/self/stat: ") + std::string(err) ;
	} else {
		std::string line;
		int i = 0;
		while (std::getline(f, line, ' ')) {
			i++;
			if (i == 39) {
				core = atoi(line.c_str());
				ret << "vcore" << Glue::colon() <<line;
			} else if (i == 1) {
				ret << "pid" << Glue::colon() <<line;
			} else if (i == 14) {
				ret << "utime" << Glue::colon() << line ;
			} else if (i == 15) {
				ret << "stime" << Glue::colon() << line ;
			} else if (i == 22) {
				ret << "starttime" << Glue::colon() << line ;
			} else if (i == 23) {
				ret << "vsize" << Glue::colon() << line ;
			}
		}
		f.close();
	}
	return std::make_pair( ret.str(), core );
}

int gpuNumber() {
#ifdef CROSS_GPU
	int dev = -1;
	cudaError_t status = cudaGetDevice(&dev);
	if (status == cudaSuccess) return dev;
	if (status == cudaErrorNoDevice) return -1;
	return -2;
#endif
	return -3;
}

JSON nodesJSON(MPI_Comm comm, bool detailed) {
	JSONarray nodes;
	JSONobject ret;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	ret << "size" << Glue::colon() << size;
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
		JSON nodejson;
		if (rank == firstrank) nodejson = nodeJSON(comm);
		nodejson = MPI_Bcast(nodejson, firstrank, comm);
		nodes << nodejson;
		i++;
		MPI_Allreduce(&wrank, &firstrank, 1, MPI_INT, MPI_MIN, comm );
		if (firstrank >= size) break;
	}
	std::pair< JSON, int > procjson = procJSON(mynode);
	int core = procjson.second;
	if (detailed) {
		JSONarray ranks;
		for (int i=0; i<size; i++) {
			JSON otherproc;
			otherproc = MPI_Bcast(procjson.first, i, comm);
			ranks << otherproc;
		}
		ret << "ranks" << Glue::colon() <<ranks.str();
	}
	int mygpu = gpuNumber();
	int* perrank = NULL;
	if (rank == 0) perrank = new int[size];
	MPI_Gather(&core, 1, MPI_INT, perrank, 1, MPI_INT, 0, comm);
	if (rank == 0) ret << "rank_to_vcore" << Glue::colon() << (JSONarray() << std::make_pair(perrank,size)).str();
	MPI_Gather(&mynode, 1, MPI_INT, perrank, 1, MPI_INT, 0, comm);
	if (rank == 0) ret << "rank_to_node" << Glue::colon() << (JSONarray() << std::make_pair(perrank,size)).str();
	MPI_Gather(&mygpu, 1, MPI_INT, perrank, 1, MPI_INT, 0, comm);
	if (rank == 0) ret << "rank_to_gpu" << Glue::colon() << (JSONarray() << std::make_pair(perrank,size)).str();
	if (rank == 0) delete[] perrank;
	ret << "nodes" << Glue::colon() <<nodes.str();
	return ret.str();
}

JSON reformatJSON(const JSON& info) {
	JSON info_formated;
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
		} else if (c == '\t') {
			continue;
		} else if (c == '\n') {
			continue;
		} else if (c == '\r') {
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


JSON stripJSON(const JSON& info) {
	JSON info_formated;
	bool in_quote = false;
	for (size_t i = 0; i < info.size(); i++) {
		char c = info[i];
		if (in_quote) {
			if (c == '"') in_quote = false;
		} else if (c == '"') {
			in_quote = true;
		} else if (c == ' ') {
			continue;
		} else if (c == '\t') {
			continue;
		} else if (c == '\n') {
			continue;
		} else if (c == '\r') {
			continue;
		}
		info_formated.push_back(c);
	}
	return info_formated;
}

JSON versionJSON(const int& major, const int& minor) {
//	JSONobject ret;
//	ret << "major" << Glue::colon() << major;
//	ret << "minor" << Glue::colon() << minor;
	JSONarray ret;
	ret <<  major << minor;
	return ret.str();
}

JSON libJSON(const std::string& name, const int& major, const int& minor) {
	JSONobject ret;
	ret << "name" << Glue::colon() << name ;
	ret << "version" << Glue::colon() << versionJSON(major, minor);
	return ret.str();
}

JSON CUDAlibJSON(const int& version) {
	int major = version/1000;
	int minor = (version - major*1000)/10;
	const std::string name = "CUDA";
	JSONobject ret;
	ret << "name" << Glue::colon() << name ;
	ret << "version" << Glue::colon() <<versionJSON(major, minor);
	return ret.str();
}

JSON compilationJSON() {
	JSONobject ret;
	JSON val;

	val = "null";
#ifdef __linux__
	val = Glue::alwaysquote("linux");
#endif
#ifdef _WIN32
	val = Glue::alwaysquote("windows");
#endif
	ret << "os" << Glue::colon() <<val;

	val = "null";
#ifdef __GNUC__
	val = libJSON("gcc",__GNUC__,__GNUC_MINOR__);
#endif
#ifdef __clang__
	val = libJSON("clang",__clang_major__,__clang_minor__);
#endif
#ifdef __EMSCRIPTEN__
	val = libJSON("emscripten",0,0);
#endif
#ifdef __MINGW32__
	val = libJSON("MinWG32",__MINGW32_MAJOR_VERSION,__MINGW32_MINOR_VERSION);
#endif
#ifdef __MINGW64__
	val = libJSON("MinWG64",__MINGW64_MAJOR_VERSION,__MINGW64_MINOR_VERSION);
#endif
	ret << "compiler" << Glue::colon() << val;

	val = "null";
#ifdef __GLIBC__
	val = libJSON("glibc",__GLIBC__,__GLIBC_MINOR__);
#endif
	ret << "glibc" << Glue::colon() <<val;

	val = "null";
#ifdef CROSS_GPU
	val = CUDAlibJSON(CUDART_VERSION);
#endif
	ret << "cuda" << Glue::colon() <<val;

	return ret.str();
}

JSON runtimeJSON() {
	JSONobject ret;
	JSON val;
	std::string name;
	int major, minor;
	JSONobject fragment;

	val = "null";
#ifdef __GLIBC__
	val = Glue::alwaysquote(gnu_get_libc_version());
#endif
	ret << "glibc" << Glue::colon() << val;

	val = "null";
#ifdef CROSS_GPU
	{
		int ver;
		cudaRuntimeGetVersion(&ver);
		val = CUDAlibJSON(ver);
	}
#endif
	ret << "cuda_runtime" << Glue::colon() <<val;
	val = "null";
#ifdef CROSS_GPU
	{
		int ver;
		cudaDriverGetVersion(&ver);
		val = CUDAlibJSON(ver);
	}
#endif
	ret << "cuda_driver" << Glue::colon() <<val;

	return ret.str();
}
