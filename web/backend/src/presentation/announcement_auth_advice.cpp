#include "presentation/announcement_auth_advice.h"

#include <drogon/drogon.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>
#include <string_view>

namespace chromaprint3d::backend {
namespace {

constexpr const char* kAnnouncementTokenHeader      = "x-announcement-token";
constexpr std::string_view kAnnouncementsPathPrefix = "/api/v1/announcements";

bool IsAnnouncementsWritePath(std::string_view path) {
    // Matches /api/v1/announcements and /api/v1/announcements/<id>
    if (path != kAnnouncementsPathPrefix &&
        path.rfind(std::string(kAnnouncementsPathPrefix) + "/", 0) != 0) {
        return false;
    }
    return true;
}

bool IsWriteMethod(drogon::HttpMethod method) {
    return method == drogon::Post || method == drogon::Delete;
}

bool ConstantTimeEquals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
    }
    return diff == 0;
}

bool IsSecureTransport(const drogon::HttpRequestPtr& req) {
    if (req->isOnSecureConnection()) return true;
    const auto& fwd = req->getHeader("x-forwarded-proto");
    if (!fwd.empty()) {
        // Lowercase compare — Nginx typically sends "https".
        std::string lowered;
        lowered.reserve(fwd.size());
        std::transform(fwd.begin(), fwd.end(), std::back_inserter(lowered),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lowered == "https") return true;
    }
    // Accept requests coming from loopback even over plain HTTP: deployment
    // always terminates TLS at Nginx so 127.0.0.1 traffic is the only legit
    // plain-HTTP source (e.g. the backend's own probes).
    if (req->peerAddr().isLoopbackIp()) return true;
    return false;
}

drogon::HttpResponsePtr BuildJsonResponse(drogon::HttpStatusCode status, const std::string& code,
                                          const std::string& message) {
    nlohmann::json body = {
        {"ok", false},
        {"error", {{"code", code}, {"message", message}}},
    };
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(status);
    resp->setBody(body.dump());
    resp->setContentTypeString("application/json; charset=utf-8");
    return resp;
}

drogon::HttpResponsePtr BuildDisabledResponse() {
    // When the feature is disabled we deliberately return 404 with a
    // generic payload so that external probes cannot distinguish a server
    // without the announcement feature from one with an unauthenticated
    // request to an existing endpoint.
    return BuildJsonResponse(drogon::k404NotFound, "not_found", "Not Found");
}

} // namespace

void RegisterAnnouncementAuthAdvice(const ServerConfig& cfg) {
    auto& app = drogon::app();

    const std::string token_copy = cfg.announcement_token;
    const bool allow_http        = cfg.announcement_allow_http;

    app.registerPreRoutingAdvice([token_copy, allow_http](const drogon::HttpRequestPtr& req,
                                                          drogon::AdviceCallback&& acb,
                                                          drogon::AdviceChainCallback&& ccb) {
        const auto& path = req->path();
        if (!IsAnnouncementsWritePath(path)) {
            ccb();
            return;
        }
        if (!IsWriteMethod(req->method())) {
            // GET / OPTIONS fall through to the controller (public read).
            ccb();
            return;
        }

        if (token_copy.empty()) {
            acb(BuildDisabledResponse());
            return;
        }

        if (!allow_http && !IsSecureTransport(req)) {
            acb(BuildJsonResponse(drogon::k403Forbidden, "insecure_transport",
                                  "Announcement writes require HTTPS"));
            return;
        }

        if (req->method() == drogon::Post) {
            const auto body = req->body();
            if (body.size() > kAnnouncementMaxBodyBytes) {
                acb(BuildJsonResponse(drogon::k413RequestEntityTooLarge, "payload_too_large",
                                      "Announcement body exceeds 16KB"));
                return;
            }
        }

        const auto provided = std::string(req->getHeader(kAnnouncementTokenHeader));
        if (provided.empty()) {
            acb(BuildJsonResponse(drogon::k401Unauthorized, "unauthorized",
                                  "Missing announcement token"));
            return;
        }
        if (!ConstantTimeEquals(provided, token_copy)) {
            acb(BuildJsonResponse(drogon::k401Unauthorized, "unauthorized",
                                  "Invalid announcement token"));
            return;
        }

        ccb();
    });
}

} // namespace chromaprint3d::backend
