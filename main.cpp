#include <stdlib.h>
#include <stdio.h>
//#include <curl/curl.h>
#include <string>
//#include <thread>
#include "glue.hpp"
#include "mpi_json_info.h"
#include "pugixml.hpp"
#include "json_to_xml.hpp"
//#include "curlCall.h"
//#include "faunadb.h"





int main (int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
    char* faunadb_token = getenv("FAUNADB_KEY");
    printf("token: %s\n",faunadb_token);
            JSONobject infoglue;
            infoglue << "MPI"  << Glue::colon() << nodesJSON(comm, true);
            JSON localinfo = localJSON();
            infoglue << "user"  << Glue::colon() << localinfo;
//            JSON geoinfo = curlAPICall("https://ipapi.co/json/");
//            infoglue << "IP"  << Glue::colon() << geoinfo;
            infoglue << "compile"  << Glue::colon() << compilationJSON();
            infoglue << "runtime"  << Glue::colon() << runtimeJSON();

            JSON info;
            info = stripJSON(infoglue.str());
	if (rank == 0) {

        std::cout << reformatJSON(info) << std::endl;
        pugi::xml_document doc;
        pugi::xml_node node = doc.append_child("Run");
        if (! JSONtoXML(info).convert(node) ) exit(-1);
        node.print(std::cout);

        //faunadb::Connection con(faunadb_token);
        //con.create_document(info, "runs");
    }

	MPI_Finalize();
}
