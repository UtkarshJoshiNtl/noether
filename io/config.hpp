#pragma once

#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace noether {

// Minimal JSON-subset parser for flat config objects:
//   { "key": <number|string|true|false|null>, ... }
// Strings may contain escaped \" \\ \n \t. Nested objects/arrays are rejected.
// Zero external dependencies by design — the engine is a research artifact.

struct Config {
    std::map<std::string, std::string> kv;

    bool has(const std::string& key) const { return kv.count(key) != 0; }

    std::string str(const std::string& key, const std::string& def = "") const {
        auto it = kv.find(key);
        return it == kv.end() ? def : it->second;
    }

    double num(const std::string& key, double def) const {
        auto it = kv.find(key);
        if (it == kv.end())
            return def;
        const std::string& s = it->second;
        const char* begin = s.c_str();
        char* end = nullptr;
        double v = std::strtod(begin, &end);
        if (end == begin || *end != '\0') {
            throw std::runtime_error("config: key '" + key + "' is not a number: " + s);
        }
        return v;
    }

    int integer(const std::string& key, int def) const {
        return static_cast<int>(num(key, static_cast<double>(def)));
    }
};

namespace detail {

inline void skip_ws(const std::string& s, std::size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
        ++i;
}

inline char expect(const std::string& s, std::size_t& i, char c) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != c) {
        throw std::runtime_error("config: expected '" + std::string(1, c) + "' at offset " +
                                 std::to_string(i));
    }
    return s[i++];
}

inline std::string parse_string(const std::string& s, std::size_t& i) {
    expect(s, i, '"');
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            switch (s[i]) {
                case 'n':
                    out += '\n';
                    break;
                case 't':
                    out += '\t';
                    break;
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                default:
                    throw std::runtime_error("config: unsupported escape");
            }
            ++i;
        } else {
            out += s[i++];
        }
    }
    if (i >= s.size())
        throw std::runtime_error("config: unterminated string");
    ++i;  // closing quote
    return out;
}

}  // namespace detail

inline Config parse_json(const std::string& text) {
    Config cfg;
    std::size_t i = 0;
    detail::skip_ws(text, i);
    detail::expect(text, i, '{');
    detail::skip_ws(text, i);
    if (i < text.size() && text[i] == '}') {
        ++i;
        return cfg;
    }
    while (true) {
        std::string key = detail::parse_string(text, i);
        detail::expect(text, i, ':');
        detail::skip_ws(text, i);
        if (i >= text.size())
            throw std::runtime_error("config: unexpected end of input");
        if (text[i] == '"') {
            cfg.kv[key] = detail::parse_string(text, i);
        } else if (text[i] == '{' || text[i] == '[') {
            throw std::runtime_error("config: only flat objects are supported (key '" + key + "')");
        } else {
            std::string tok;
            while (i < text.size() && text[i] != ',' && text[i] != '}' && text[i] != ' ' &&
                   text[i] != '\t' && text[i] != '\n' && text[i] != '\r') {
                tok += text[i++];
            }
            if (tok.empty())
                throw std::runtime_error("config: empty value for key '" + key + "'");
            cfg.kv[key] = tok;
        }
        detail::skip_ws(text, i);
        if (i >= text.size())
            throw std::runtime_error("config: unexpected end of input");
        if (text[i] == ',') {
            ++i;
            continue;
        }
        if (text[i] == '}') {
            ++i;
            break;
        }
        throw std::runtime_error("config: expected ',' or '}' at offset " + std::to_string(i));
    }
    return cfg;
}

inline Config load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("config: cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_json(ss.str());
}

}  // namespace noether
