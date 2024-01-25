#include <stdio.h>
#include "pugixml.hpp"
#include "glue.hpp"
#include "json_to_xml.hpp"

bool JSONtoXML::str(std::string& val) {
    if (!expect('"')) return false;
    bool in_escape = false;
    while (now() != '\0') {
        if (in_escape) {
            in_escape = false;
        } else if (now() == '\\') {
            in_escape = true;
        } else if (now() == '"') {
            return expect('"');
        }
        val.push_back(now());
        i++;
    }
    return false;
}

bool JSONtoXML::rest(std::string& val) {
    const std::string fin = "{}[],\"\0";
    while (fin.find(now()) == std::string::npos) {
        val.push_back(now());
        i++;
    }
    return true;
}

bool JSONtoXML::arr(pugi::xml_node& node, std::string name) {
    Glue text_val(" ");
    bool is_text = true;
    if (!expect('[')) return false;
    while (now() != '\0') {
        if (now() == '{') {
            is_text = false;
            pugi::xml_node child = node.append_child(name.c_str());
            if (!obj(child)) return false;
        } else if (now() == '[') {
            return false;
        } else if (now() == '"') {
            std::string val;
            if (!str(val)) return false;
            text_val << val;
        } else {
            std::string val;
            if (!rest(val)) return false;
            text_val << val;
        }
        if (expect(']')) {
            if (is_text) {
                node.append_attribute(name.c_str()).set_value(text_val.c_str());
            } else {
                if (!text_val.is_empty()) return false;
            }
            return true;
        }
        if (!expect(',')) return false;
    }
    return false;
}

bool JSONtoXML::obj(pugi::xml_node& node) {
    if (!expect('{')) return false;
    if (expect('}')) return true;
    while (now() != '\0') {
        std::string name;
        if (!str(name)) return false;
        if (!expect(':')) return false;
        if (now() == '{') {
            pugi::xml_node child = node.append_child(name.c_str());
            if (!obj(child)) return false;
        } else if (now() == '[') {
            if (!arr(node,name)) return false;
        } else if (now() == '"') {
            std::string val;
            if (!str(val)) return false;
            node.append_attribute(name.c_str()).set_value(val.c_str());
        } else {
            std::string val;
            if (!rest(val)) return false;
            node.append_attribute(name.c_str()).set_value(val.c_str());
        }
        if (expect('}')) return true;
        if (!expect(',')) return false;
    }
    return false;
}

JSONtoXML::JSONtoXML(std::string data_): data(data_), i(0) {};

bool JSONtoXML::convert(pugi::xml_node& node) {    
    i=0;
    if (!obj(node)) {
        printf("Something went wrong in JSON to XML convertion!\n");
        printf("%s >>> %c <<< %s\n",data.substr(0,i).c_str(), now(), data.substr(i+1,data.length()).c_str());
        return false;
    }
    return true;
}
