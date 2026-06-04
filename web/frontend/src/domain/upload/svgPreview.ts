const SVG_NAMESPACE = 'http://www.w3.org/2000/svg'

const PREVIEW_STROKE_OVERRIDE = `
svg, svg * {
  stroke: none !important;
  stroke-width: 0 !important;
}
`

function hasParserError(doc: Document): boolean {
  return doc.getElementsByTagName('parsererror').length > 0
}

function createStrokeOverrideStyle(doc: Document): Element {
  const style = doc.createElementNS(SVG_NAMESPACE, 'style')
  style.setAttribute('data-chromaprint3d-preview', 'hide-stroke')
  style.textContent = PREVIEW_STROKE_OVERRIDE
  return style
}

export async function createSvgPreviewWithoutStroke(svg: Blob): Promise<Blob> {
  const text = await svg.text()
  const parser = new DOMParser()
  const doc = parser.parseFromString(text, 'image/svg+xml')
  const root = doc.documentElement

  if (!root || root.localName.toLowerCase() !== 'svg' || hasParserError(doc)) {
    return svg
  }

  root.insertBefore(createStrokeOverrideStyle(doc), root.firstChild)
  const serialized = new XMLSerializer().serializeToString(doc)
  return new Blob([serialized], { type: 'image/svg+xml' })
}
