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

std::string gpuJSON() {
	Glue ret(", ","{ ", " }");
#ifdef CROSS_GPU
	cudaError_t status;
	int ngpus;
	Glue gpus(", ","[ ", " ]");
	status = cudaGetDeviceCount(&ngpus);
	if (status == cudaErrorNoDevice) {
		ngpus = 0;
		status = cudaSuccess;
	}
	if (status != cudaSuccess) {
		ret << (Glue() << "\"error\": \"cudaGetDeviceCount: " << cudaGetErrorString(status) << "\"").str();
	} else {
		for (int i=0; i<ngpus; i++) {
			Glue gpu(", ","{ ", " }");
			cudaDeviceProp prop;
			status = cudaGetDeviceProperties(&prop, i);
			if (status != cudaSuccess) {
				gpu << (Glue() << "\"error\": \"cudaGetDeviceProperties: " << cudaGetErrorString(status) << "\"").str();
			} else {
				gpu << (Glue() << "\"name\": \"" << prop.name << "\"").str();  
				gpu << (Glue() << "\"totalGlobalMem\": " << prop.totalGlobalMem).str();
				gpu << (Glue() << "\"sharedMemPerBlock\": " << prop.sharedMemPerBlock).str();
				gpu << (Glue() << "\"version\": " << versionJSON(prop.major,prop.minor)).str();
				gpu << (Glue() << "\"clockRate\": " << prop.clockRate).str();
				gpu << (Glue() << "\"multiProcessorCount\": " << prop.multiProcessorCount).str();
				gpu << (Glue() << "\"ECCEnabled\": " << (prop.ECCEnabled ? "true" : "false")).str();
			}
			gpus << gpu.str();
		}
		ret << (Glue() << "\"gpus:\": " << ngpus).str();
		ret << "\"gpu:\": " + gpus.str();
	}
#else
	ret << "\"gpus\": 0";
	ret << "\"warning\": \"compiled without CUDA\"";
#endif
	return ret.str();
}

std::string cpuJSON() {
	Glue ret(", ","{ ", " }");
	Glue physicalid(", ","[ ", " ]");
	Glue coreid(", ","[ ", " ]");
	Glue cpus(", ","[ ", " ]");
	int ncores=0;
	int ncpu=0;
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
	ret << (Glue() << "\"vcores\": " << ncores).str();
	ret << (Glue() << "\"cpus\": " << ncpu).str();
	ret << "\"vcore_to_cpu\": " + physicalid.str();
	ret << "\"vcore_to_core\": " + coreid.str();
	ret << "\"cpu\": " + cpus.str();
	return ret.str();
}

std::string nodeJSON(MPI_Comm comm) {
	Glue ret(", ","{ ", " }");
	ret << "\"name\": \"" + nodeName(comm) + "\"";
	ret << "\"cpu\": " + cpuJSON();
	ret << "\"gpu\": " + gpuJSON();
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
	int mygpu = gpuNumber();
	int* perrank = NULL;
	if (rank == 0) perrank = new int[size];
	MPI_Gather(&core, 1, MPI_INT, perrank, 1, MPI_INT, 0, comm);
	if (rank == 0) ret << "\"rank_to_vcore\": " + (Glue(", ", "[ ", " ]") << std::make_pair(perrank,size)).str();
	MPI_Gather(&mynode, 1, MPI_INT, perrank, 1, MPI_INT, 0, comm);
	if (rank == 0) ret << "\"rank_to_node\": " + (Glue(", ", "[ ", " ]") << std::make_pair(perrank,size)).str();
	MPI_Gather(&mygpu, 1, MPI_INT, perrank, 1, MPI_INT, 0, comm);
	if (rank == 0) ret << "\"rank_to_gpu\": " + (Glue(", ", "[ ", " ]") << std::make_pair(perrank,size)).str();
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


std::string stripJSON(const std::string& info) {
	std::string info_formated;
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

std::string versionJSON(const int& major, const int& minor) {
	Glue ret(", ", "{ ", " }");
	ret << (Glue() << "\"major\": " << major).str();
	ret << (Glue() << "\"minor\": " << minor).str();
	return ret.str();
}

std::string libJSON(const std::string& name, const int& major, const int& minor) {
	Glue ret(", ", "{ ", " }");
	ret << "\"name\": \"" + name + "\"";
	ret << "\"version\": " + versionJSON(major, minor);
	return ret.str();
}

std::string CUDAlibJSON(const int& version) {
	int major = version/1000;
	int minor = (version - major*1000)/10;
	const std::string name = "CUDA";
	Glue ret(", ", "{ ", " }");
	ret << "\"name\": \"" + name + "\"";
	ret << "\"version\": " + versionJSON(major, minor);
	return ret.str();
}

std::string compilationJSON() {
	Glue ret(", ", "{ ", " }");
	std::string val;

	val = "null";
#ifdef __linux__
	val = "\"linux\"";
#endif
#ifdef _WIN32
	val = "\"windows\"";
#endif
	ret << "\"os\": " + val;

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
	ret << "\"compiler\": " + val;

#ifdef __GLIBC__
	val = libJSON("glibc",__GLIBC__,__GLIBC_MINOR__);
#endif
	ret << "\"glibc\": " + val;

#ifdef CROSS_GPU
	val = CUDAlibJSON(CUDART_VERSION);
#endif
	ret << "\"cuda\": " + val;

	return ret.str();
}

std::string runtimeJSON() {
	Glue ret(", ", "{ ", " }");
	std::string val,name;
	int major, minor;
	Glue fragment(", ", "{ ", " }");

	val = "null";
#ifdef __GLIBC__
	val = gnu_get_libc_version();
	val = "\"" + val + "\"";
#endif
	ret << "\"glibc\": " + val;

	val = "null";
#ifdef CROSS_GPU
	{
		int ver;
		cudaRuntimeGetVersion(&ver);
		val = CUDAlibJSON(ver);
	}
#endif
	ret << "\"cuda_runtime\": " + val;
	val = "null";
#ifdef CROSS_GPU
	{
		int ver;
		cudaDriverGetVersion(&ver);
		val = CUDAlibJSON(ver);
	}
#endif
	ret << "\"cuda_driver\": " + val;

	return ret.str();
}
