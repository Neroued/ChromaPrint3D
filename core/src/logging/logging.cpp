#include "chromaprint3d/logging.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>

namespace ChromaPrint3D {

void InitLogging(spdlog::level::level_enum level) {
    auto sink   = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("chromaprint3d", sink);

    logger->set_level(level);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    logger->flush_on(spdlog::level::warn);

    spdlog::set_default_logger(logger);
    spdlog::set_level(level);
}

spdlog::level::level_enum ParseLogLevel(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    if (s == "trace") { return spdlog::level::trace; }
    if (s == "debug") { return spdlog::level::debug; }
    if (s == "info") { return spdlog::level::info; }
    if (s == "warn" || s == "warning") { return spdlog::level::warn; }
    if (s == "error" || s == "err") { return spdlog::level::err; }
    if (s == "off") { return spdlog::level::off; }
    return spdlog::level::info;
}

} // namespace ChromaPrint3D
