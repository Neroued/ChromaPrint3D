import type { LabColor } from '../types'

function gammaToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

function linearToGamma(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055
}

const D65_X = 0.95047
const D65_Y = 1.0
const D65_Z = 1.08883

function labF(t: number): number {
  return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116
}

function labFInv(t: number): number {
  return t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787
}

export function hexToLab(hex: string): LabColor {
  const raw = hex.startsWith('#') ? hex.slice(1) : hex
  const r = parseInt(raw.slice(0, 2), 16) / 255
  const g = parseInt(raw.slice(2, 4), 16) / 255
  const b = parseInt(raw.slice(4, 6), 16) / 255

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

  return `#${rr.toString(16).padStart(2, '0')}${gg.toString(16).padStart(2, '0')}${bb.toString(16).padStart(2, '0')}`
}
