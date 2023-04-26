#ifndef CURLCALL_H
#define CURLCALL_H

#include <string>

int function_pt(char *text, size_t size, size_t nmemb, void *ptr);
std::string curlAPICall(const std::string& url, const std::string& payload = "");

#endif