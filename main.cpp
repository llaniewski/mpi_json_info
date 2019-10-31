#include <stdio.h>
#include <mpi.h>
#include "mpi_json_info.h"


int main ( int argc, char * argv[] )
{
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	std::string info = nodesJSON(comm, false);
	if (rank == 0) {
		printf("%s\n", reformatJSON(info).c_str());
	}
	MPI_Finalize();
}
