#include <curl/curl.h>
#include <string>
#include <thread>
#include "glue.hpp"
#include "mpi_json_info.h"

int function_pt(char *text, size_t size, size_t nmemb, void *ptr) {
    std::string * ret = (std::string *) ptr;
    std::string str((char*) text, nmemb*size);
    (*ret) += str;
    return size*nmemb;
}

std::string curlAPICall(const std::string& url, const std::string& payload = "") {
    CURL *curl;
    CURLcode res;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (curl == NULL) {
        fprintf(stderr, "curl_easy_init() failed\n");
        return "null";
    }
    printf("CURL: URL: %s\n", url.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcrp/0.1");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, function_pt);
    std::string ret;
    void * ptr = &(ret);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, ptr);
    if (payload == "") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "GET");
    } else {
        printf("CURL: Payload: %s\n", payload.c_str());
        struct curl_slist *headers = NULL;
        curl_slist_append(headers, "Accept: application/json");
        curl_slist_append(headers, "Content-Type: application/json");
        curl_slist_append(headers, "charsets: utf-8");
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    }    
    res = curl_easy_perform(curl);
    if(res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        return "null";
    }
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    return ret;
}


namespace faunadb {

namespace q {

JSON wrap(const JSON& info) {
	JSON info_formated;
	bool in_escape = false;
	bool in_quote = false;
	for (size_t i = 0; i < info.size(); i++) {
		char c = info[i];
		if (in_quote) {
		    if (in_escape) {
		    
		    } else {
		        if (c == '\\') {
		            in_escape = true;
                        } else if (c == '"') {
                            in_quote = false;
                        }
                    }
		} else if (c == '"') {
			in_quote = true;
		} else if (c == '{') {
			info_formated += "{\"object\": ";
		} else if (c == '}') {
			info_formated += "}";
        }
		info_formated.push_back(c);
	}
	return info_formated;
}

JSON object(const JSON& data) {
    JSONobject ret;
    ret << "object" << Glue::colon() << data;
    return ret.str();
}

JSON data(const JSON& data) {
    JSONobject ret;
    ret << "data" << Glue::colon() << data;
    return object(ret.str());
}

JSON create(const JSON& collection, const JSON& params) {
    JSONobject ret;
    ret << "create" << Glue::colon() << collection;
    ret << "params" << Glue::colon() << params;
    return ret.str();
}

JSON collection(const std::string& data) {
    JSONobject ret;
    ret << "collection" << Glue::colon() << data;
    return ret.str();
}

}

class Connection {
    std::string token;
    std::string protocol;
    std::string domain;
    std::string url;
public:
    Connection(const std::string& token_, const std::string& protocol_="https", const std::string& domain_="db.fauna.com" )
    : token(token_), protocol(protocol_), domain(domain_) { url = protocol + "://" + token + "@" + domain + "/"; }
     
    int post(const JSON& payload) {
        JSON ret;
        ret = curlAPICall(url, payload);
        printf("faunadb result: %s\n", ret.c_str());
        return 0;
    }

    int create_document(JSON document, JSON collection) {
        JSON payload = 
            q::create(
                q::collection(collection),
                q::data(
                    q::wrap(document)
                )
            );
        return post(payload);
    }
};

}

void faunasend(JSON nodesinfo) {
            JSON info;
//            Glue infoglue(", ","{ "," }");
            JSONobject infoglue;
            infoglue << "MPI"  << Glue::colon() << nodesinfo;
            JSON localinfo = localJSON();
            infoglue << "local"  << Glue::colon() << localinfo;
            JSON geoinfo = curlAPICall("https://ipapi.co/json/");
            infoglue << "IP"  << Glue::colon() << geoinfo;
            info = stripJSON(infoglue.str());
            printf("%s\n", reformatJSON(info).c_str());

using namespace faunadb;
        JSON payload = 
            q::create(
                q::collection("a\na"),
                q::data(
                    q::wrap(info)
                )
            );
            printf("payload:\n%s\n", payload.c_str());
            printf("payload:\n%s\n", reformatJSON(payload).c_str());
  
/*
            faunadb::Connection *con = new faunadb::Connection("TOKEN");
            con->create_document(info, "stats");
            delete con;
*/
}

int main (int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	JSON nodesinfo = nodesJSON(comm, false);
    
	if (rank == 0) {
	    faunasend(nodesinfo);
        }
	MPI_Finalize();
}
