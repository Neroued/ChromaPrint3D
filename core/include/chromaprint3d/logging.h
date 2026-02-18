/// \file logging.h
/// \brief Logging initialization and utilities.

#pragma once

#include <spdlog/spdlog.h>

namespace ChromaPrint3D {

/// Initializes the global logger with default format, level, and color sink.
/// Call once at the start of main() before any logging.
/// \param level Log level (default: info)
void InitLogging(spdlog::level::level_enum level = spdlog::level::info);

/// Parses a log level string to spdlog level enum.
/// \param str Log level string ("trace", "debug", "info", "warn", "error", "off")
/// \return Parsed log level, or spdlog::level::info if unrecognized
spdlog::level::level_enum ParseLogLevel(const std::string& str);

} // namespace ChromaPrint3D
