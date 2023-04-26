#include "json.hpp"

JSON reformatJSON(const JSON& info) {
	JSON info_formated;
	int ind = 0;
	bool in_quote = false;
	bool in_escape = false;
	int nl_after = -1;
	int nl_before = -1;
	bool space_after = false;
	bool hold_nl = false;
	for (size_t i = 0; i < info.size(); i++) {
		char c = info[i];
		if (in_quote) {
			if (in_escape) {
				in_escape = false;
			} else if (c == '\\') {
				in_escape = true;
			} else if (c == '"') {
				in_quote = false;
			}
		} else if (c == '"') {
			in_quote = true;
		} else if (c == ' ') {
			continue;
		} else if (c == '\t') {
			continue;
		} else if (c == '\n') {
			continue;
		} else if (c == '\r') {
			continue;
		} else if (c == ':') {
			space_after = true;
		} else if (c == '{') {
			if (hold_nl) {
				nl_before = ind;
				hold_nl = false;
			}
			ind++;
			nl_after = ind;
		} else if (c == '}') {
			ind--;
			nl_before = ind;
		} else if (c == '[') {
			ind++;
			hold_nl = ind;
		} else if (c == ']') {
			ind--;
			if (hold_nl) hold_nl = false; else nl_before = ind;
		} else if (c == ',') {
			if (hold_nl) space_after = true; else nl_after = ind;
		}
		if (nl_before >= 0) {
			info_formated.push_back('\n');
			for (int j = 0; j<nl_before*2; j++) info_formated.push_back(' ');
			nl_before = -1;
		}
		info_formated.push_back(c);
		if (nl_after >= 0) {
			info_formated.push_back('\n');
			for (int j = 0; j<nl_after*2; j++) info_formated.push_back(' ');
			nl_after = -1;
		}
		if (space_after) {
			info_formated.push_back(' ');
			space_after = false;
		}
	}
	return info_formated;
}


JSON stripJSON(const JSON& info) {
	JSON info_formated;
	bool in_quote = false;
	bool in_escape = false;
	for (size_t i = 0; i < info.size(); i++) {
		char c = info[i];
		if (in_quote) {
			if (in_escape) {
				in_escape = false;
			} else if (c == '\\') {
				in_escape = true;
			} else if (c == '"') {
				in_quote = false;
			}
		} else if (c == '"') {
			in_quote = true;
		} else if (c == ' ') {
			continue;
		} else if (c == '\t') {
			continue;
		} else if (c == '\n') {
			continue;
		} else if (c == '\r') {
			continue;
		}
		info_formated.push_back(c);
	}
	return info_formated;
}
