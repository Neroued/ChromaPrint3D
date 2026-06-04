// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { createSvgPreviewWithoutStroke } from './svgPreview'

class MockElement {
  localName: string
  textContent: string | null = null
  firstChild: MockElement | null = null
  private attributes: Record<string, string> = {}

  constructor(localName: string) {
    this.localName = localName
  }

  setAttribute(name: string, value: string) {
    this.attributes[name] = value
  }

  insertBefore(child: MockElement) {
    this.firstChild = child
  }

  serialize(): string {
    const attrs = Object.entries(this.attributes)
      .map(([key, value]) => `${key}="${value}"`)
      .join(' ')
    return `<${this.localName}${attrs ? ` ${attrs}` : ''}>${this.textContent ?? ''}</${
      this.localName
    }>`
  }
}

class MockDocument {
  readonly documentElement: MockElement
  readonly source: string
  private readonly parserError: boolean

  constructor(source: string, parserError = false) {
    this.source = source
    this.parserError = parserError
    this.documentElement = new MockElement(parserError ? 'parsererror' : 'svg')
  }

  getElementsByTagName(name: string): MockElement[] {
    return this.parserError && name === 'parsererror' ? [this.documentElement] : []
  }

  createElementNS(_namespace: string, name: string): MockElement {
    return new MockElement(name)
  }
}

const originalDomParser = globalThis.DOMParser
const originalXmlSerializer = globalThis.XMLSerializer

describe('svgPreview', () => {
  beforeEach(() => {
    Object.defineProperty(globalThis, 'DOMParser', {
      configurable: true,
      writable: true,
      value: class {
        parseFromString(text: string): MockDocument {
          return new MockDocument(text, !text.trim().startsWith('<svg'))
        }
      },
    })
    Object.defineProperty(globalThis, 'XMLSerializer', {
      configurable: true,
      writable: true,
      value: class {
        serializeToString(doc: MockDocument): string {
          const injected = doc.documentElement.firstChild?.serialize() ?? ''
          return doc.source.replace(/<svg([^>]*)>/, `<svg$1>${injected}`)
        }
      },
    })
  })

  afterEach(() => {
    Object.defineProperty(globalThis, 'DOMParser', {
      configurable: true,
      writable: true,
      value: originalDomParser,
    })
    Object.defineProperty(globalThis, 'XMLSerializer', {
      configurable: true,
      writable: true,
      value: originalXmlSerializer,
    })
  })

  it('为 SVG 预览插入去描边样式', async () => {
    const input = new Blob(
      [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path fill="#fff" stroke="#000" stroke-width="1" d="M0 0h10v10H0z"/></svg>',
      ],
      { type: 'image/svg+xml' },
    )

    const output = await createSvgPreviewWithoutStroke(input)
    const text = await output.text()

    expect(output).not.toBe(input)
    expect(text).toContain('data-chromaprint3d-preview="hide-stroke"')
    expect(text).toContain('stroke: none !important')
    expect(text).toContain('stroke-width: 0 !important')
    expect(text).toContain('stroke="#000"')
  })

  it('解析失败时回退原始 Blob', async () => {
    const input = new Blob(['not svg'], { type: 'image/svg+xml' })

    const output = await createSvgPreviewWithoutStroke(input)

    expect(output).toBe(input)
  })
})
