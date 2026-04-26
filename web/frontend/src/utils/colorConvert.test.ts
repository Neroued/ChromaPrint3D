import { describe, expect, it } from 'vitest'
import { hexToLab, labToHex } from './colorConvert'

describe('colorConvert hex round-trip', () => {
  // Per plan §4.6: random 1000 hex round-trip with ±1 byte tolerance,
  // matching the SrgbU8::FromHex/ToHex contract on the C++ side.
  it('round-trips 1000 random hex values within ±1 byte', () => {
    const seed = 1
    let state = seed
    const rng = () => {
      state = (state * 1664525 + 1013904223) >>> 0
      return state
    }

    const toHex = (n: number) => n.toString(16).padStart(2, '0')
    const buildHex = () => {
      const r = rng() & 0xff
      const g = rng() & 0xff
      const b = rng() & 0xff
      return `#${toHex(r)}${toHex(g)}${toHex(b)}`.toUpperCase()
    }

    let maxDelta = 0
    for (let i = 0; i < 1000; ++i) {
      const original = buildHex()
      const lab = hexToLab(original)
      const back = labToHex(lab)

      const r1 = parseInt(original.slice(1, 3), 16)
      const g1 = parseInt(original.slice(3, 5), 16)
      const b1 = parseInt(original.slice(5, 7), 16)
      const r2 = parseInt(back.slice(1, 3), 16)
      const g2 = parseInt(back.slice(3, 5), 16)
      const b2 = parseInt(back.slice(5, 7), 16)

      const dr = Math.abs(r1 - r2)
      const dg = Math.abs(g1 - g2)
      const db = Math.abs(b1 - b2)
      maxDelta = Math.max(maxDelta, dr, dg, db)

      expect(dr).toBeLessThanOrEqual(1)
      expect(dg).toBeLessThanOrEqual(1)
      expect(db).toBeLessThanOrEqual(1)
    }
    expect(maxDelta).toBeLessThanOrEqual(1)
  })

  it('produces L=0 for black and L=100 for white', () => {
    const black = hexToLab('#000000')
    expect(black.L).toBeCloseTo(0, 4)
    const white = hexToLab('#FFFFFF')
    expect(white.L).toBeCloseTo(100, 1)
  })

  it('outputs uppercase #RRGGBB matching SrgbU8::ToHex format', () => {
    expect(labToHex({ L: 0, a: 0, b: 0 })).toMatch(/^#[0-9A-F]{6}$/)
    expect(labToHex({ L: 100, a: 0, b: 0 })).toMatch(/^#[0-9A-F]{6}$/)
  })
})

describe('colorConvert hex parsing strictness', () => {
  it('accepts #RGB short form', () => {
    const lab = hexToLab('#F80')
    expect(lab.L).toBeGreaterThan(50)
  })

  it('accepts 0x prefix', () => {
    const labA = hexToLab('0xFF8000')
    const labB = hexToLab('#FF8000')
    expect(labA.L).toBeCloseTo(labB.L, 5)
  })

  it('accepts bare hex without prefix', () => {
    const lab = hexToLab('FF8000')
    expect(lab.L).toBeGreaterThan(40)
  })

  it('rejects 8-digit RGBA #RRGGBBAA', () => {
    expect(() => hexToLab('#FF000080')).toThrow()
  })

  it('rejects 8-digit RGBA without prefix', () => {
    expect(() => hexToLab('FF000080')).toThrow()
  })

  it('rejects invalid lengths', () => {
    expect(() => hexToLab('#FF')).toThrow()
    expect(() => hexToLab('#FFFF')).toThrow()
    expect(() => hexToLab('#FFFFF')).toThrow()
    expect(() => hexToLab('#FFFFFFFF F')).toThrow()
  })

  it('rejects non-hex characters', () => {
    expect(() => hexToLab('#GGGGGG')).toThrow()
    expect(() => hexToLab('#ZZZ')).toThrow()
  })
})
