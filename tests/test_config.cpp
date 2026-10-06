#include <cstdio>
#include <fstream>
#include <string>

#include "io/config.hpp"

// JSON-subset config parser: values, defaults, overrides, and rejection of bad input.

using namespace noether;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

template <typename F>
void check_throws(F&& fn, const char* what) {
    try {
        fn();
        std::printf("FAIL: no exception: %s\n", what);
        ++failures;
    } catch (const std::exception&) {
        // expected
    }
}

void test_values() {
    Config c = parse_json(
        "{"
        "\"integrator\": \"yoshida4\","
        "\"dt\": 2.5,"
        "\"n\": 64,"
        "\"out\": \"a\\\\b\\\"c\""
        "}");
    check(c.str("integrator") == "yoshida4", "string value");
    check(c.num("dt", 0.0) == 2.5, "double value");
    check(c.integer("n", 0) == 64, "int value");
    check(c.str("out") == "a\\b\"c", "escaped string");
    check(!c.has("missing"), "has() on missing key");
    check(c.str("missing", "def") == "def", "str default");
    check(c.num("missing", 9.0) == 9.0, "num default");
    check(c.integer("missing", -1) == -1, "integer default");
    check(c.num("missing", 9.0) == 9.0, "num default unchanged");
}

void test_whitespace_and_empty() {
    Config c = parse_json("  { \n\t \"dt\" :\r\n 10 } ");
    check(c.num("dt", 0.0) == 10.0, "whitespace tolerated");
    Config e = parse_json("{}");
    check(e.kv.empty(), "empty object");
}

void test_rejections() {
    check_throws([] { parse_json("{\"a\": {\"b\": 1}}"); }, "nested object rejected");
    check_throws([] { parse_json("{\"a\": [1,2]}"); }, "array rejected");
    check_throws([] { parse_json("{\"a\": }"); }, "empty value rejected");
    check_throws([] { parse_json("{\"a\": 1"); }, "unterminated object rejected");
    check_throws([] { parse_json("{\"a\": \"x"); }, "unterminated string rejected");
    check_throws([] { parse_json("not json"); }, "non-JSON rejected");
    Config c = parse_json("{\"a\": 1.0}");
    check(c.num("a", 0.0) == 1.0, "num() parses plain number token");
    Config s = parse_json("{\"a\": true}");
    check_throws([&] { s.num("a", 0.0); }, "num() on non-number token");
}

void test_load_file() {
    const char* path = "test_config_tmp.json";
    {
        std::ofstream f(path);
        f << "{\"dt\": 5, \"integrator\": \"verlet\"}";
    }
    Config c = load_config(path);
    check(c.num("dt", 0.0) == 5.0, "load_config number");
    check(c.str("integrator") == "verlet", "load_config string");
    std::remove(path);
    check_throws([&] { load_config("does_not_exist.json"); }, "missing file rejected");
}

}  // namespace

int main() {
    test_values();
    test_whitespace_and_empty();
    test_rejections();
    test_load_file();
    if (failures) {
        std::printf("test_config: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_config: OK\n");
    return 0;
}
