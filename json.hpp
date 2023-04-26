#ifndef JSON_HPP
#define JSON_HPP

#include "glue.hpp"

class JSONobject : public Glue { public: JSONobject() : Glue(",","{","}", true) {} };
class JSONarray  : public Glue { public: JSONarray() : Glue(",","[","]", true) {} };
typedef Glue::neverquote JSON;

JSON reformatJSON(const JSON&);
JSON stripJSON(const JSON&);

#endif //JSON_HPP