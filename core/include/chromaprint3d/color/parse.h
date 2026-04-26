#pragma once

/// \file color/parse.h
/// \brief Hex parsing (`SrgbU8::FromHex`), named-color lookup, fallback
///        palette, and `ResolveColorLiteral` — the **single** place that
///        turns a user-provided palette literal into an `SrgbU8`.

#include "chromaprint3d/color/types.h"

#include <optional>
#include <string>
#include <string_view>

namespace ChromaPrint3D {

/// Look up a named color by canonical name (case-insensitive, non-alphanumeric
/// characters are stripped: `"Bambu Green"` → `"bambugreen"`).
/// Returns `std::nullopt` if no exact match.
std::optional<SrgbU8> NamedColorLookup(std::string_view name);

/// Stable-hash fallback color from an arbitrary literal. Always returns a
/// non-grey color (gives palette UI deterministic visuals when a literal
/// has no known mapping).
SrgbU8 HashFallbackColor(std::string_view key);

/// Built-in fallback palette by index. Wraps modulo length. Used by
/// `print_profile.cpp` and `FilamentConfig::fallback_palette`.
SrgbU8 FallbackColorByIndex(int index);

/// **The** entry point for resolving an arbitrary user palette literal:
///   1. `SrgbU8::FromHex(literal)` succeeds → return.
///   2. Normalize key; empty → mid-grey `{127,127,127}`.
///   3. `NamedColorLookup(key)` matches → return.
///   4. Fuzzy substring match against canonical name table → return.
///   5. `HashFallbackColor(key)` (deterministic, key-derived).
///
/// Pure function: input → output. Channel-out-of-range fallback semantics
/// (e.g. `recipe_map.cpp`'s "channel index out of palette") are the
/// caller's responsibility — call `ResolveColorLiteral` only after that
/// guard.
SrgbU8 ResolveColorLiteral(std::string_view literal);

} // namespace ChromaPrint3D
