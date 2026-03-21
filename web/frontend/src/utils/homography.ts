/**
 * 4-point homography (DLT method) for projecting board-space coordinates
 * to image-space coordinates given 4 corresponding point pairs.
 *
 * Solves H such that  dst = H * src  (in homogeneous coordinates).
 * src: 4 canonical fiducial positions in board space
 * dst: 4 detected/adjusted fiducial positions in image space
 */

type Point2D = [number, number]

/**
 * Compute 3x3 homography matrix from 4 source->destination point pairs.
 * Returns a flat 9-element array [h00,h01,h02, h10,h11,h12, h20,h21,h22].
 */
export function computeHomography(src: Point2D[], dst: Point2D[]): number[] {
  const n = src.length
  const A: number[][] = []
  const b: number[] = []

  for (let i = 0; i < n; i++) {
    const [sx, sy] = src[i]!
    const [dx, dy] = dst[i]!
    A.push([-sx, -sy, -1, 0, 0, 0, sx * dx, sy * dx])
    b.push(-dx)
    A.push([0, 0, 0, -sx, -sy, -1, sx * dy, sy * dy])
    b.push(-dy)
  }

  const h = solveLinear8x8(A, b)
  return [h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!, 1]
}

/**
 * Project a point through the homography matrix.
 */
export function projectPoint(H: number[], p: Point2D): Point2D {
  const w = H[6]! * p[0] + H[7]! * p[1] + H[8]!
  if (Math.abs(w) < 1e-12) return [0, 0]
  const x = (H[0]! * p[0] + H[1]! * p[1] + H[2]!) / w
  const y = (H[3]! * p[0] + H[4]! * p[1] + H[5]!) / w
  return [x, y]
}

/**
 * Compute grid sample point positions in image space.
 * Returns array of {row, col, x, y} for each patch center.
 */
export function computeSamplePositions(
  corners: Point2D[],
  gridRows: number,
  gridCols: number,
  boardGeometry: {
    boardW: number
    boardH: number
    margin: number
    tile: number
    gap: number
    offset: number
  },
): { row: number; col: number; x: number; y: number }[] {
  const { boardW, boardH, margin, tile, gap, offset } = boardGeometry

  const canonical: Point2D[] = [
    [offset, offset],
    [boardW - offset, offset],
    [boardW - offset, boardH - offset],
    [offset, boardH - offset],
  ]

  const H = computeHomography(canonical, corners)
  const results: { row: number; col: number; x: number; y: number }[] = []

  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      const bx = margin + c * (tile + gap) + tile / 2
      const by = margin + r * (tile + gap) + tile / 2
      const [ix, iy] = projectPoint(H, [bx, by])
      results.push({ row: r, col: c, x: ix, y: iy })
    }
  }

  return results
}

function solveLinear8x8(A: number[][], b: number[]): number[] {
  const cols = 8
  const rows = A.length
  const aug: number[][] = []
  for (let i = 0; i < rows; i++) {
    const row = A[i]!
    aug.push([...row, b[i]!])
  }

  for (let col = 0; col < cols; col++) {
    let maxRow = col
    let maxVal = Math.abs(aug[col]![col]!)
    for (let row = col + 1; row < rows; row++) {
      const v = Math.abs(aug[row]![col]!)
      if (v > maxVal) {
        maxVal = v
        maxRow = row
      }
    }
    if (maxRow !== col) {
      const tmp = aug[col]!
      aug[col] = aug[maxRow]!
      aug[maxRow] = tmp
    }
    const pivotRow = aug[col]!
    const pivot = pivotRow[col]!
    if (Math.abs(pivot) < 1e-12) continue
    for (let j = col; j <= cols; j++) pivotRow[j] = pivotRow[j]! / pivot
    for (let row = 0; row < rows; row++) {
      if (row === col) continue
      const curRow = aug[row]!
      const factor = curRow[col]!
      for (let j = col; j <= cols; j++) curRow[j] = curRow[j]! - factor * pivotRow[j]!
    }
  }

  const x: number[] = []
  for (let i = 0; i < cols; i++) x.push(aug[i]![cols]!)
  return x
}
