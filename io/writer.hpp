#pragma once

#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace noether {

// CSV output with 17-significant-digit round-trip precision.
// The engine writes raw data only; figures are produced by analysis/ scripts.

struct CsvWriter {
    std::ofstream f;

    CsvWriter(const std::string& path, const std::string& header) {
        f.open(path, std::ios::out | std::ios::trunc);
        if (!f)
            throw std::runtime_error("output: cannot open " + path);
        f << header << '\n';
    }

    void row(std::initializer_list<double> cols) {
        bool first = true;
        for (double c : cols) {
            if (!first)
                f << ',';
            first = false;
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", c);
            f << buf;
        }
        f << '\n';
    }
};

}  // namespace noether
