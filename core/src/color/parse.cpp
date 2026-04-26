/// \file core/src/color/parse.cpp
/// \brief Hex parsing, named-color lookup, fallback colors, and the unified
///        `ResolveColorLiteral` entry — replaces the five disparate hex
///        parsers and the `detail::PaletteColorLiteralToBgr` helper.

#include "chromaprint3d/color/parse.h"

#include "chromaprint3d/color/types.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ChromaPrint3D {

namespace {

bool ParseHexNibble(char c, std::uint8_t& nibble) {
    if (c >= '0' && c <= '9') {
        nibble = static_cast<std::uint8_t>(c - '0');
        return true;
    }
    if (c >= 'a' && c <= 'f') {
        nibble = static_cast<std::uint8_t>(10 + c - 'a');
        return true;
    }
    if (c >= 'A' && c <= 'F') {
        nibble = static_cast<std::uint8_t>(10 + c - 'A');
        return true;
    }
    return false;
}

bool ParseHexByte(char hi, char lo, std::uint8_t& out) {
    std::uint8_t h = 0, l = 0;
    if (!ParseHexNibble(hi, h) || !ParseHexNibble(lo, l)) return false;
    out = static_cast<std::uint8_t>((h << 4) | l);
    return true;
}

std::string TrimAscii(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) { --end; }
    return std::string(value.substr(begin, end - begin));
}

std::string NormalizeKeyImpl(std::string_view literal) {
    std::string out;
    out.reserve(literal.size());
    for (unsigned char c : literal) {
        if (std::isalnum(c) != 0) out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

const std::unordered_map<std::string, SrgbU8>& NamedColorTableImpl() {
    static const std::unordered_map<std::string, SrgbU8> table = {
        {"white", {255, 255, 255}},     {"black", {0, 0, 0}},
        {"red", {255, 0, 0}},           {"green", {0, 255, 0}},
        {"blue", {0, 0, 255}},          {"yellow", {255, 255, 0}},
        {"cyan", {0, 255, 255}},        {"magenta", {255, 0, 255}},
        {"orange", {255, 165, 0}},      {"purple", {128, 0, 128}},
        {"pink", {255, 192, 203}},      {"gray", {128, 128, 128}},
        {"grey", {128, 128, 128}},      {"brown", {165, 42, 42}},
        {"gold", {255, 215, 0}},        {"silver", {192, 192, 192}},
        {"bambugreen", {34, 139, 34}},
    };
    return table;
}

constexpr std::array<const char*, 16> kCanonicalNames = {
    "white",  "black",  "red",  "green", "blue", "yellow", "cyan", "magenta",
    "orange", "purple", "pink", "gray",  "grey", "brown",  "gold", "silver"};

} // namespace

// ── SrgbU8 hex API ──────────────────────────────────────────────────────────

std::optional<SrgbU8> SrgbU8::FromHex(std::string_view hex) {
    std::string trimmed = TrimAscii(hex);
    std::string_view s  = trimmed;

    if (!s.empty() && s.front() == '#') s.remove_prefix(1);
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s.remove_prefix(2);

    if (s.empty()) return std::nullopt;

    // Strict: only payload length 3 or 6. 8-char (RGBA) is rejected — RGBA
    // must go through `neroued_3mf::Color::FromHex`.
    if (s.size() == 3) {
        std::uint8_t rn = 0, gn = 0, bn = 0;
        if (!ParseHexNibble(s[0], rn) || !ParseHexNibble(s[1], gn) ||
            !ParseHexNibble(s[2], bn)) {
            return std::nullopt;
        }
        return SrgbU8{static_cast<std::uint8_t>(rn * 17),
                      static_cast<std::uint8_t>(gn * 17),
                      static_cast<std::uint8_t>(bn * 17)};
    }

    if (s.size() == 6) {
        std::uint8_t r = 0, g = 0, b = 0;
        if (!ParseHexByte(s[0], s[1], r)) return std::nullopt;
        if (!ParseHexByte(s[2], s[3], g)) return std::nullopt;
        if (!ParseHexByte(s[4], s[5], b)) return std::nullopt;
        return SrgbU8{r, g, b};
    }

    return std::nullopt;
}

std::string SrgbU8::ToHex(const SrgbU8& c) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "#%02X%02X%02X", c.r, c.g, c.b);
    return std::string(buf);
}

// ── Named lookup / fallback ─────────────────────────────────────────────────

std::optional<SrgbU8> NamedColorLookup(std::string_view name) {
    const std::string key = NormalizeKeyImpl(name);
    if (key.empty()) return std::nullopt;
    const auto& table = NamedColorTableImpl();
    auto it           = table.find(key);
    if (it == table.end()) return std::nullopt;
    return it->second;
}

SrgbU8 HashFallbackColor(std::string_view key) {
    const std::uint32_t h =
        static_cast<std::uint32_t>(std::hash<std::string_view>{}(key));
    std::uint8_t b = static_cast<std::uint8_t>(64u + (h & 0x7Fu));
    std::uint8_t g = static_cast<std::uint8_t>(64u + ((h >> 8) & 0x7Fu));
    std::uint8_t r = static_cast<std::uint8_t>(64u + ((h >> 16) & 0x7Fu));
    if (std::max({r, g, b}) - std::min({r, g, b}) < 16) {
        r = static_cast<std::uint8_t>(std::min<int>(255, r + 48));
    }
    return SrgbU8{r, g, b};
}

namespace {

// Small built-in fallback palette: 16 visually-distinct colors. Index is
// taken modulo length, so callers can blindly pass channel index.
constexpr SrgbU8 kBuiltinFallbackPalette[16] = {
    {0xC0, 0x39, 0x2B}, {0x27, 0xAE, 0x60}, {0x29, 0x80, 0xB9}, {0xF1, 0xC4, 0x0F},
    {0x8E, 0x44, 0xAD}, {0xE6, 0x7E, 0x22}, {0x16, 0xA0, 0x85}, {0x2C, 0x3E, 0x50},
    {0xE7, 0x4C, 0x3C}, {0x2E, 0xCC, 0x71}, {0x34, 0x98, 0xDB}, {0xF3, 0x9C, 0x12},
    {0x9B, 0x59, 0xB6}, {0xD3, 0x54, 0x00}, {0x1A, 0xBC, 0x9C}, {0x7F, 0x8C, 0x8D},
};

} // namespace

SrgbU8 FallbackColorByIndex(int index) {
    constexpr int kLen = sizeof(kBuiltinFallbackPalette) / sizeof(kBuiltinFallbackPalette[0]);
    if (kLen <= 0) return SrgbU8{127, 127, 127};
    int wrapped = index % kLen;
    if (wrapped < 0) wrapped += kLen;
    return kBuiltinFallbackPalette[wrapped];
}

SrgbU8 ResolveColorLiteral(std::string_view literal) {
    // 1. Direct hex parse.
    if (auto hex = SrgbU8::FromHex(literal); hex) return *hex;

    // 2. Normalized key, empty → mid-grey.
    const std::string key = NormalizeKeyImpl(literal);
    if (key.empty()) return SrgbU8{127, 127, 127};

    // 3. Exact named lookup.
    if (auto named = NamedColorLookup(key); named) return *named;

    // 4. Fuzzy substring match against canonical names.
    for (const char* token : kCanonicalNames) {
        if (key.find(token) != std::string::npos) {
            const auto& table = NamedColorTableImpl();
            auto it           = table.find(token);
            if (it != table.end()) return it->second;
        }
    }

    // 5. Stable hashed fallback.
    return HashFallbackColor(key);
}

} // namespace ChromaPrint3D
