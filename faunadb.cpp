#include <stdio.h>
#include "faunadb.h"
#include "curlCall.h"

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

Connection::Connection(const std::string& token_, const std::string& protocol_, const std::string& domain_)
    : token(token_), protocol(protocol_), domain(domain_) {
        url = protocol + "://" + token + "@" + domain + "/";
}
     
int Connection::post(const JSON& payload) {
    JSON ret;
    ret = curlAPICall(url, payload);
    printf("faunadb result: %s\n", ret.c_str());
    return 0;
}

int Connection::create_document(JSON document, JSON collection) {
    JSON payload = 
        q::create(
            q::collection(collection),
            q::data(
                q::wrap(document)
            )
        );
    return post(payload);
}

}
