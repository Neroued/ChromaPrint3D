import type { LabColor } from '../types'

function gammaToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

const D65_X = 0.95047
const D65_Y = 1.0
const D65_Z = 1.08883

function labF(t: number): number {
  return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116
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
