#include "presentation/cors_advice.h"

#include <drogon/drogon.h>
#include <nlohmann/json.hpp>

namespace chromaprint3d::backend {
namespace {

constexpr const char* kSessionHeader = "X-ChromaPrint3D-Session";

bool IsApiRequest(const drogon::HttpRequestPtr& req) {
    const auto& path = req->path();
    return path == "/api/v1" || path.rfind("/api/v1/", 0) == 0;
}

bool IsOriginAllowed(const drogon::HttpRequestPtr& req, const ServerConfig& cfg) {
    const auto origin = req->getHeader("origin");
    if (cfg.cors_origin.empty()) return true;
    if (origin.empty()) return true;
    return origin == cfg.cors_origin;
}

void ApplyCorsHeaders(const drogon::HttpRequestPtr& req, const drogon::HttpResponsePtr& resp,
                      const ServerConfig& cfg) {
    const auto origin = req->getHeader("origin");
    resp->addHeader("Vary", "Origin");
    if (cfg.cors_origin.empty()) {
        if (!origin.empty()) {
            resp->addHeader("Access-Control-Allow-Origin", origin);
        } else {
            resp->addHeader("Access-Control-Allow-Origin", "*");
        }
    } else if (!origin.empty() && origin == cfg.cors_origin) {
        resp->addHeader("Access-Control-Allow-Origin", cfg.cors_origin);
    }

    resp->addHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    resp->addHeader("Access-Control-Allow-Headers", std::string("Content-Type, ") + kSessionHeader);
    resp->addHeader("Access-Control-Allow-Credentials", "true");
    resp->addHeader("Access-Control-Expose-Headers",
                    std::string(kSessionHeader) + ", X-Region-Map-Width, X-Region-Map-Height");
    resp->addHeader("Access-Control-Max-Age", "86400");
}

drogon::HttpResponsePtr BuildForbiddenOriginResponse(const drogon::HttpRequestPtr& req,
                                                     const ServerConfig& cfg) {
    nlohmann::json body = {
        {"ok", false},
        {"error", {{"code", "forbidden_origin"}, {"message", "Origin not allowed"}}},
    };

    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k403Forbidden);
    resp->setBody(body.dump());
    resp->setContentTypeString("application/json; charset=utf-8");
    ApplyCorsHeaders(req, resp, cfg);
    return resp;
}

} // namespace

void RegisterCorsAdvice(const ServerConfig& cfg) {
    auto& app = drogon::app();

    app.registerPreRoutingAdvice([cfg](const drogon::HttpRequestPtr& req,
                                       drogon::AdviceCallback&& acb,
                                       drogon::AdviceChainCallback&& ccb) {
        if (!IsApiRequest(req)) {
            ccb();
            return;
        }

        if (!IsOriginAllowed(req, cfg)) {
            acb(BuildForbiddenOriginResponse(req, cfg));
            return;
        }

        if (req->method() == drogon::Options) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k204NoContent);
            ApplyCorsHeaders(req, resp, cfg);
            acb(resp);
            return;
        }

        ccb();
    });

    app.registerPreSendingAdvice(
        [cfg](const drogon::HttpRequestPtr& req, const drogon::HttpResponsePtr& resp) {
            if (!IsApiRequest(req)) return;
            ApplyCorsHeaders(req, resp, cfg);
        });
}

} // namespace chromaprint3d::backend
