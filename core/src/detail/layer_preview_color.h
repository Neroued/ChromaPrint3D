#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace ChromaPrint3D::detail {

inline std::string TrimAscii(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) { --end; }
    return value.substr(begin, end - begin);
}

inline bool ParseHexNibble(char c, uint8_t& nibble) {
    if (c >= '0' && c <= '9') {
        nibble = static_cast<uint8_t>(c - '0');
        return true;
    }
    if (c >= 'a' && c <= 'f') {
        nibble = static_cast<uint8_t>(10 + c - 'a');
        return true;
    }
    if (c >= 'A' && c <= 'F') {
        nibble = static_cast<uint8_t>(10 + c - 'A');
        return true;
    }
    return false;
}

inline bool ParseHexByte(char hi, char lo, uint8_t& out) {
    uint8_t h = 0;
    uint8_t l = 0;
    if (!ParseHexNibble(hi, h) || !ParseHexNibble(lo, l)) return false;
    out = static_cast<uint8_t>((h << 4) | l);
    return true;
}

inline bool TryParseHexColor(const std::string& literal, uint8_t& r, uint8_t& g, uint8_t& b) {
    std::string s = TrimAscii(literal);
    if (s.size() >= 1 && s[0] == '#') s = s.substr(1);
    if (s.size() >= 2 && (s[0] == '0') && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);

    if (s.size() == 3) {
        uint8_t rn = 0, gn = 0, bn = 0;
        if (!ParseHexNibble(s[0], rn) || !ParseHexNibble(s[1], gn) || !ParseHexNibble(s[2], bn)) {
            return false;
        }
        r = static_cast<uint8_t>(rn * 17);
        g = static_cast<uint8_t>(gn * 17);
        b = static_cast<uint8_t>(bn * 17);
        return true;
    }

    if (s.size() != 6) return false;
    if (!ParseHexByte(s[0], s[1], r)) return false;
    if (!ParseHexByte(s[2], s[3], g)) return false;
    if (!ParseHexByte(s[4], s[5], b)) return false;
    return true;
}

inline std::string NormalizeColorKey(const std::string& literal) {
    std::string out;
    out.reserve(literal.size());
    for (unsigned char c : literal) {
        if (std::isalnum(c) != 0) out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

inline const std::unordered_map<std::string, cv::Vec3b>& NamedColorTable() {
    static const std::unordered_map<std::string, cv::Vec3b> table = {
        {"white", cv::Vec3b(255, 255, 255)},    {"black", cv::Vec3b(0, 0, 0)},
        {"red", cv::Vec3b(0, 0, 255)},          {"green", cv::Vec3b(0, 255, 0)},
        {"blue", cv::Vec3b(255, 0, 0)},         {"yellow", cv::Vec3b(0, 255, 255)},
        {"cyan", cv::Vec3b(255, 255, 0)},       {"magenta", cv::Vec3b(255, 0, 255)},
        {"orange", cv::Vec3b(0, 165, 255)},     {"purple", cv::Vec3b(128, 0, 128)},
        {"pink", cv::Vec3b(203, 192, 255)},     {"gray", cv::Vec3b(128, 128, 128)},
        {"grey", cv::Vec3b(128, 128, 128)},     {"brown", cv::Vec3b(42, 42, 165)},
        {"gold", cv::Vec3b(0, 215, 255)},       {"silver", cv::Vec3b(192, 192, 192)},
        {"bambugreen", cv::Vec3b(34, 139, 34)},
    };
    return table;
}

inline cv::Vec3b FallbackColorFromKey(const std::string& key) {
    const std::uint32_t h = static_cast<std::uint32_t>(std::hash<std::string>{}(key));
    uint8_t b             = static_cast<uint8_t>(64u + (h & 0x7Fu));
    uint8_t g             = static_cast<uint8_t>(64u + ((h >> 8) & 0x7Fu));
    uint8_t r             = static_cast<uint8_t>(64u + ((h >> 16) & 0x7Fu));
    if (std::max({r, g, b}) - std::min({r, g, b}) < 16) {
        r = static_cast<uint8_t>(std::min<int>(255, r + 48));
    }
    return cv::Vec3b(b, g, r);
}

inline cv::Vec3b PaletteColorLiteralToBgr(const std::string& literal) {
    uint8_t r = 0, g = 0, b = 0;
    if (TryParseHexColor(literal, r, g, b)) return cv::Vec3b(b, g, r);

    const std::string key = NormalizeColorKey(literal);
    if (key.empty()) return cv::Vec3b(127, 127, 127);

    const auto& table = NamedColorTable();
    auto it           = table.find(key);
    if (it != table.end()) return it->second;

    static const std::string canonical[] = {
        "white",  "black",  "red",  "green", "blue", "yellow", "cyan", "magenta",
        "orange", "purple", "pink", "gray",  "grey", "brown",  "gold", "silver"};
    for (const auto& token : canonical) {
        if (key.find(token) != std::string::npos) {
            auto exact = table.find(token);
            if (exact != table.end()) return exact->second;
        }
    }
    return FallbackColorFromKey(key);
}

} // namespace ChromaPrint3D::detail
