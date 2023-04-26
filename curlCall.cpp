#include "curlCall.h"
#include <curl/curl.h>
#include <string>

int function_pt(char *text, size_t size, size_t nmemb, void *ptr) {
    std::string * ret = (std::string *) ptr;
    std::string str((char*) text, nmemb*size);
    (*ret) += str;
    return size*nmemb;
}

std::string curlAPICall(const std::string& url, const std::string& payload) {
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
