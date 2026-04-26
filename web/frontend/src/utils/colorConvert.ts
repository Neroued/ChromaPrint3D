import type { LabColor } from '../types'

// Project D65 sRGB ↔ Lab math, byte-equivalent (within floating-point tolerance)
// to the C++ `chromaprint3d/color/conversions.h` and Python
// `modeling/core/color_space.py::linear_rgb_to_lab_d65` implementations.
//
// Precise constants — do **not** revert to decimal approximations like
// `0.008856` / `7.787` / `0.206893`. The exact closed-form is:
//   delta      = 6 / 29
//   delta^3    = (6/29)^3        // breakpoint for the LabF piecewise function
//   3*delta^2  = 3 * (6/29)^2    // slope of the linear segment (= 7.78704...)
// Drift between approximate decimals and the exact form is sub-ΔE in the
// general case, but breaks bitwise parity with the C++/Python paths.

function gammaToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

function linearToGamma(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055
}

const D65_X = 0.95047
const D65_Y = 1.0
const D65_Z = 1.08883

const LAB_DELTA = 6 / 29
const LAB_DELTA_CUBED = LAB_DELTA * LAB_DELTA * LAB_DELTA
const LAB_LINEAR_SLOPE = 3 * LAB_DELTA * LAB_DELTA
const LAB_OFFSET = 4 / 29

function labF(t: number): number {
  return t > LAB_DELTA_CUBED ? Math.cbrt(t) : t / LAB_LINEAR_SLOPE + LAB_OFFSET
}

function labFInv(t: number): number {
  return t > LAB_DELTA ? t * t * t : LAB_LINEAR_SLOPE * (t - LAB_OFFSET)
}

/// Parse hex `#RRGGBB` / `#RGB` / `0xRRGGBB` / bare into Lab. Throws if the
/// payload length is not 3 or 6 — matching `SrgbU8::FromHex`'s strict
/// rejection of 8-digit RGBA inputs.
export function hexToLab(hex: string): LabColor {
  let s = hex.trim()
  if (s.startsWith('#')) s = s.slice(1)
  else if (s.startsWith('0x') || s.startsWith('0X')) s = s.slice(2)

  let r: number
  let g: number
  let b: number
  if (s.length === 3) {
    const rn = parseInt(s.charAt(0), 16)
    const gn = parseInt(s.charAt(1), 16)
    const bn = parseInt(s.charAt(2), 16)
    if (Number.isNaN(rn) || Number.isNaN(gn) || Number.isNaN(bn)) {
      throw new Error(`hexToLab: invalid hex digits in '${hex}'`)
    }
    r = (rn * 17) / 255
    g = (gn * 17) / 255
    b = (bn * 17) / 255
  } else if (s.length === 6) {
    r = parseInt(s.slice(0, 2), 16) / 255
    g = parseInt(s.slice(2, 4), 16) / 255
    b = parseInt(s.slice(4, 6), 16) / 255
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
      throw new Error(`hexToLab: invalid hex digits in '${hex}'`)
    }
  } else {
    throw new Error(`hexToLab: payload length must be 3 or 6, got ${s.length} ('${hex}')`)
  }

  const lr = gammaToLinear(r)
  const lg = gammaToLinear(g)
  const lb = gammaToLinear(b)

  const x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / D65_X
  const y = (0.2126729 * lr + 0.7151522 * lg + 0.072175 * lb) / D65_Y
  const z = (0.0193339 * lr + 0.119192 * lg + 0.9503041 * lb) / D65_Z

  const fx = labF(x)
  const fy = labF(y)
  const fz = labF(z)

  return {
    L: 116 * fy - 16,
    a: 500 * (fx - fy),
    b: 200 * (fy - fz),
  }
}

/// Format Lab back into `#RRGGBB` (uppercase, matching C++
/// `SrgbU8::ToHex`).
export function labToHex(lab: LabColor): string {
  const fy = (lab.L + 16) / 116
  const fx = lab.a / 500 + fy
  const fz = fy - lab.b / 200

  const x = labFInv(fx) * D65_X
  const y = labFInv(fy) * D65_Y
  const z = labFInv(fz) * D65_Z

  const lr = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
  const lg = -0.969266 * x + 1.8760108 * y + 0.041556 * z
  const lb = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

  const clamp = (v: number) => Math.max(0, Math.min(1, v))
  const toByte = (v: number) => Math.round(clamp(linearToGamma(v)) * 255)

  const rr = toByte(lr)
  const gg = toByte(lg)
  const bb = toByte(lb)

  // Uppercase to match SrgbU8::ToHex output (`#RRGGBB`).
  return `#${rr.toString(16).padStart(2, '0')}${gg.toString(16).padStart(2, '0')}${bb.toString(16).padStart(2, '0')}`.toUpperCase()
}
