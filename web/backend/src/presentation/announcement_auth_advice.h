#pragma once

#include "config/server_config.h"

#include <cstddef>

namespace chromaprint3d::backend {

// Maximum accepted POST body for /api/v1/announcements (16 KB).
// Chosen deliberately small — announcements are plain-text summaries,
// not user artifacts. Larger bodies are rejected with 413 to prevent
// abuse of the public announcement endpoint by an attacker who gained
// access to the announcement token.
inline constexpr std::size_t kAnnouncementMaxBodyBytes = 16 * 1024;

// Registers a Drogon pre-routing advice that guards POST and DELETE on
// /api/v1/announcements. Behavior summary:
//   * GET  is always allowed (public read path).
//   * POST/DELETE without a configured --announcement-token → 404
//     (not 401) so servers without the feature enabled do not leak
//     route existence.
//   * POST/DELETE over plain HTTP (X-Forwarded-Proto != https) are
//     rejected with 403 unless --announcement-allow-http is enabled
//     for local debugging.
//   * Token comparison is constant-time.
//   * POST bodies larger than kAnnouncementMaxBodyBytes are rejected
//     with 413.
//
// No dev/debug bypass exists on purpose. This repository is open source
// and the security model assumes the auth flow is public; secrecy lives
// in the token plus a mandatory Nginx IP allowlist in deployment.
void RegisterAnnouncementAuthAdvice(const ServerConfig& cfg);

} // namespace chromaprint3d::backend
