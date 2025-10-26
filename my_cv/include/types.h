#pragma once

#include <dirent.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// 단순 디렉터리 나열
static inline int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) return -1;

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
            file_names.emplace_back(p_file->d_name);
        }
    }
    closedir(p_dir);
    return 0;
}

static inline std::string trim_leading_whitespace(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

static inline std::string to_string_with_precision(const float a_value, const int n = 2) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

static inline int read_labels(const std::string& labels_filename, std::unordered_map<int, std::string>& labels_map) {
    std::ifstream file(labels_filename);
    if (!file.is_open()) return -1;

    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        line = trim_leading_whitespace(line);
        labels_map[index++] = line;
    }
    file.close();
    return 0;
}
