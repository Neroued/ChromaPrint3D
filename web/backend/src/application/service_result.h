#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace chromaprint3d::backend {

struct ServiceResult {
    bool ok         = false;
    int status_code = 500;
    std::string code;
    std::string message;
    nlohmann::json data = nlohmann::json::object();

    static ServiceResult Success(int status, nlohmann::json d) {
        ServiceResult r;
        r.ok          = true;
        r.status_code = status;
        r.data        = std::move(d);
        return r;
    }

    static ServiceResult Error(int status, std::string c, std::string m) {
        ServiceResult r;
        r.ok          = false;
        r.status_code = status;
        r.code        = std::move(c);
        r.message     = std::move(m);
        return r;
    }
};

} // namespace chromaprint3d::backend
