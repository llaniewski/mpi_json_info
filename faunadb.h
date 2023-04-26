#ifndef FAUNADB_H
#define FAUNADB_H

#include <string>
#include "json.hpp"
namespace faunadb {
    namespace q {
        JSON wrap(const JSON& info);
        JSON object(const JSON& data);
        JSON data(const JSON& data);
        JSON create(const JSON& collection, const JSON& params);
        JSON collection(const std::string& data);
    }

    class Connection {
        std::string token;
        std::string protocol;
        std::string domain;
        std::string url;
    public:
        Connection(const std::string& token_, const std::string& protocol_="https", const std::string& domain_="db.fauna.com" );
        int post(const JSON& payload);
        int create_document(JSON document, JSON collection);
    };
}

#endif // FAUNADB_H