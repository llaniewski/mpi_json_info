#ifndef GLUE_H
#define GLUE_H

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

class Glue {
public:
    class alwaysquote : public std::string {
        public:
            alwaysquote(const std::string& str_=""): std::string(str_) {}
    };
    class  neverquote : public std::string {
        public:
            neverquote(const std::string& str_=""): std::string(str_) {}
            neverquote(const char str_[]): std::string(str_) {}
            neverquote(const alwaysquote& str_): std::string(quote(str_)) {}
    };
    class  separator : public std::string { public:  separator(const std::string& str_=""): std::string(str_) {} };
    static neverquote quote(std::string str) {
        std::string ret;
        ret.push_back('"');
	for (size_t i = 0; i < str.size(); i++) {
		char c = str[i];
		if (c == '"') {
                    ret.push_back('\\');
                    ret.push_back('"');
                } else if (c == '\n') {
                    ret.push_back('\\');
                    ret.push_back('n');
                } else if (c == '\t') {
                    ret.push_back('\\');
                    ret.push_back('t');
                } else {
                    ret.push_back(c);
                }
        }
        ret.push_back('"');		
        return ret;
    }
private:
    std::stringstream s;
    std::string sep;
    neverquote val;
    std::string begin;
    std::string end;
    bool empty;
    bool use_quote;
public:
    static inline const separator colon() { return separator(":"); }
        
    inline Glue(std::string sep_="", std::string begin_="", std::string end_="", bool use_quote_=false) : sep(sep_), begin(begin_), end(end_), use_quote(use_quote_) {
        s << std::setprecision(14);
        s << std::scientific;
        empty = true;
    }
    inline Glue& operator () (std::string sep_ = "") {
        s.str(std::string());
        sep = sep_;
        empty = true;
        return *this;
    }
    inline Glue& clear() {
        s.str(std::string());
        empty = true;
        return *this;
    }
    template <class T> inline Glue& glue(const T& t) {
        s << t;
        return *this;
    }
    template <class T> inline Glue& add(const T& t) {
        if (sep == "" || empty)
            s << t;
        else
            s << sep << t;
        empty = false;
        return *this;
    }
    template <class T> inline Glue& operator<< (const T& t) {
        return this->add(t);
    }
    inline Glue& operator<< (const std::string& t) {
        if (use_quote) {
            return this->add(quote(t));
        } else {
            return this->add(t);
        }        
    }
    inline Glue& operator<< (const separator& t) {
        empty=true;
        return this->glue(t);
    }
    inline Glue& operator<< (const char t[]) {
        return (*this) << std::string(t);
    }
    inline Glue& operator<< (const neverquote& t) {
        return this->add(t);
    }
    inline Glue& operator<< (const alwaysquote& t) {
        return this->add(quote(t));
    }
    inline Glue& operator<< (Glue& t) {
        return this->add(t.str());
    }
    template <class T> inline Glue& operator<< (const std::pair<T*, int>& t) {
        for (int i=0; i<t.second; i++) (*this) << t.first[i];
        return *this;
    }
    inline const neverquote& str (){
        val = begin + s.str() + end;
        return val;
    }
    inline const char* c_str () {
        return this->str().c_str();
    }
    inline operator const char* () {
        return this->str().c_str();
    }
};

class JSONobject : public Glue { public: JSONobject() : Glue(",","{","}", true) {} };
class JSONarray  : public Glue { public: JSONarray() : Glue(",","[","]", true) {} };
typedef Glue::neverquote JSON;


#endif
