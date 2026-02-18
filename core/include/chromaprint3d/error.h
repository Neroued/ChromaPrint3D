#pragma once

/// \file error.h
/// \brief Typed error hierarchy for ChromaPrint3D.
///
/// All public functions throw subclasses of ChromaPrint3D::Error instead of
/// plain std::runtime_error, so callers can catch specific categories.

#include "export.h"

#include <stdexcept>
#include <string>

namespace ChromaPrint3D {

/// Error categories returned by Error::code().
enum class ErrorCode : int {
    Ok = 0,
    InvalidInput,   ///< Caller supplied invalid arguments or data.
    IOError,        ///< File or stream I/O failure.
    FormatError,    ///< Data format / parsing error (JSON, 3MF, etc.).
    ConfigMismatch, ///< Incompatible configuration (channel count, layer height, ...).
    NoValidCandidate, ///< Color matching found no usable candidate.
    InternalError,  ///< Logic error inside the library.
};

/// Base exception for all ChromaPrint3D errors.
class CHROMAPRINT3D_API Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& msg)
        : std::runtime_error(msg), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

/// File / stream I/O failure.
class CHROMAPRINT3D_API IOError : public Error {
public:
    explicit IOError(const std::string& msg)
        : Error(ErrorCode::IOError, msg) {}
};

/// Invalid input arguments or data.
class CHROMAPRINT3D_API InputError : public Error {
public:
    explicit InputError(const std::string& msg)
        : Error(ErrorCode::InvalidInput, msg) {}
};

/// Data format / parsing failure (JSON, 3MF, ...).
class CHROMAPRINT3D_API FormatError : public Error {
public:
    explicit FormatError(const std::string& msg)
        : Error(ErrorCode::FormatError, msg) {}
};

/// Incompatible configuration across modules.
class CHROMAPRINT3D_API ConfigError : public Error {
public:
    explicit ConfigError(const std::string& msg)
        : Error(ErrorCode::ConfigMismatch, msg) {}
};

/// Color matching found no usable candidate.
class CHROMAPRINT3D_API MatchError : public Error {
public:
    explicit MatchError(const std::string& msg)
        : Error(ErrorCode::NoValidCandidate, msg) {}
};

} // namespace ChromaPrint3D
