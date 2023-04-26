
#include "pugixml.hpp"

class JSONtoXML {
    std::string data;
    size_t i;
    inline char now() {
        if (i < data.size()) return data[i];
        return '\0';
    }
    inline bool expect(char c) {
        if (now() != c) return false;
        i++; return true;
    }
    bool str(std::string& val);
    bool rest(std::string& val);
    bool arr(pugi::xml_node& node, std::string name);
    bool obj(pugi::xml_node& node);
public:
    JSONtoXML(std::string data_);
    bool convert(pugi::xml_node& node);
};
