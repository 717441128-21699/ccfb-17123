// 红漆标尺竿检测：严格红像素定位中心列 -> 红/白漆段实际像素跨度测宽
import { rgbToHsv } from './image';
import type { PoleObservation } from '../recon/reconstruct';

export function detectPole(images: ImageData[]): PoleObservation {
  const edges: PoleObservation['edges'] = images.map((img) => {
    const { width: w, height: h } = img;
    // 1) 严格红像素
    const redCols = new Int32Array(w);
    const redRows: number[] = [];
    let redCount = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const r = img.data[i * 4];
        const g = img.data[i * 4 + 1];
        const b = img.data[i * 4 + 2];
        const hsv = rgbToHsv(r, g, b);
        const isRed = (hsv.h < 16 || hsv.h > 344) && hsv.s > 0.5 && hsv.v > 0.25
          && r > g * 1.4 && r > b * 1.4;
        if (isRed) {
          redCols[x]++;
          redRows.push(y);
          redCount++;
        }
      }
    }
    let centerX = -1;
    let best = 0;
    for (let x = 0; x < w; x++) {
      if (redCols[x] > best) {
        best = redCols[x];
        centerX = x;
      }
    }
    if (centerX < 0 || best < 3) return null;

    const yTop = Math.min(...redRows);
    const yBottom = Math.max(...redRows);

    // 2) 宽度直接取每行红像素的实际列跨度（红漆是竿上最特异、最可靠的标记）。
    // 亚像素：红像素不足整像素宽时，跨度至少反映真实占用，多帧统计进一步收敛。
    const lefts: number[] = [];
    const rights: number[] = [];
    const ys: number[] = [];
    for (let y = Math.max(0, yTop - 2); y <= Math.min(h - 1, yBottom + 2); y += 1) {
      let l = -1;
      let r = -1;
      for (let dx = -4; dx <= 4; dx++) {
        const x = centerX + dx;
        if (x < 0 || x >= w) continue;
        const i = y * w + x;
        const rv = img.data[i * 4];
        const gv = img.data[i * 4 + 1];
        const bv = img.data[i * 4 + 2];
        const hsv = rgbToHsv(rv, gv, bv);
        // 只计纯红漆像素，避免白段/邻近竹竿把宽度撑大
        const isRed = (hsv.h < 16 || hsv.h > 344) && hsv.s > 0.5 && rv > gv * 1.4;
        if (isRed) {
          if (l < 0) l = x;
          r = x;
        }
      }
      if (l >= 0 && r - l >= 0 && r - l <= 8) {
        // 单像素宽物体：全宽按 1px 计（半宽 0.5），多列时取实际跨度
        lefts.push(l - (r === l ? 0.5 : 0));
        rights.push(r + (r === l ? 0.5 : 1));
        ys.push(y);
      }
    }
    if (lefts.length < 3) return null;
    return {
      left: median(lefts),
      right: median(rights),
      y: median(ys),
    };
  });
  return { edges };
}

function median(a: number[]): number {
  const s = a.slice().sort((x, y) => x - y);
  return s[s.length >> 1];
}
