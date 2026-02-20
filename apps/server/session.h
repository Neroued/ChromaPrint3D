#pragma once

#include "chromaprint3d/color_db.h"
#include "http_utils.h"

#include <httplib/httplib.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <mutex>
#include <random>
#include <regex>
#include <string>
#include <unordered_map>

using namespace ChromaPrint3D;

inline std::string GenerateUUID() {
    static thread_local std::mt19937_64 rng([] {
        std::random_device rd;
        return rd();
    }());
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t a = dist(rng);
    uint64_t b = dist(rng);
    char buf[37];
    std::snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
                  static_cast<uint32_t>(a >> 32), static_cast<uint16_t>(a >> 16),
                  static_cast<uint16_t>((a & 0xFFFF) | 0x4000),
                  static_cast<uint16_t>((b >> 48) | 0x8000),
                  static_cast<unsigned long long>(b & 0xFFFFFFFFFFFFULL));
    return std::string(buf);
}

static constexpr int kMaxSessionColorDBs = 10;

struct UserSession {
    std::string token;
    std::unordered_map<std::string, ColorDB> color_dbs;
    std::chrono::steady_clock::time_point last_access;
};

struct SessionManager {
    std::mutex mtx;
    std::unordered_map<std::string, UserSession> sessions;
    int ttl_seconds = 3600;

    UserSession& GetOrCreate(const std::string& token) {
        std::lock_guard<std::mutex> lock(mtx);
        auto& session      = sessions[token];
        session.token       = token;
        session.last_access = std::chrono::steady_clock::now();
        return session;
    }

    UserSession* Find(const std::string& token) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = sessions.find(token);
        if (it == sessions.end()) { return nullptr; }
        it->second.last_access = std::chrono::steady_clock::now();
        return &it->second;
    }

    void CleanExpired() {
        std::lock_guard<std::mutex> lock(mtx);
        auto now = std::chrono::steady_clock::now();
        for (auto it = sessions.begin(); it != sessions.end();) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_access)
                    .count();
            if (elapsed > ttl_seconds) {
                spdlog::info("Session expired: {}", it->first.substr(0, 8));
                it = sessions.erase(it);
            } else {
                ++it;
            }
        }
    }
};

inline bool IsValidDBName(const std::string& name) {
    if (name.empty() || name.size() > 64) { return false; }
    static const std::regex pattern("^[a-zA-Z0-9_]+$");
    return std::regex_match(name, pattern);
}

inline std::string GetSessionToken(const httplib::Request& req) {
    if (!req.has_header("Cookie")) { return ""; }
    std::string cookie = req.get_header_value("Cookie");
    static const std::regex re("(?:^|;\\s*)session=([a-f0-9\\-]+)");
    std::smatch m;
    if (std::regex_search(cookie, m, re)) { return m[1].str(); }
    return "";
}

inline std::string EnsureSession(const httplib::Request& req, httplib::Response& res,
                                 SessionManager& session_mgr) {
    std::string token = GetSessionToken(req);
    if (token.empty()) {
        token            = GenerateUUID();
        auto cookie_attr = IsCrossOriginMode() ? "; HttpOnly; SameSite=None; Secure; Path=/"
                                               : "; HttpOnly; SameSite=Strict; Path=/";
        res.set_header("Set-Cookie", "session=" + token + cookie_attr);
        spdlog::info("New session created: {}", token.substr(0, 8));
    }
    session_mgr.GetOrCreate(token);
    return token;
}
