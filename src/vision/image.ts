// 图像处理基础：灰度、缩放、形态学、Sobel、HSV
import { clamp } from '../core/linalg';

export interface GrayImage {
  w: number;
  h: number;
  /** 亮度 0~255 */
  data: Float32Array;
}

export interface GradImage {
  w: number;
  h: number;
  gx: Float32Array;
  gy: Float32Array;
  mag: Float32Array;
  /** 梯度方向（弧度，0~2π） */
  ori: Float32Array;
}

export function toGray(img: ImageData): GrayImage {
  const { width: w, height: h, data } = img;
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    out[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  return { w, h, data: out };
}

/** 区域均值缩放到目标宽高（双线性） */
export function resizeGray(src: GrayImage, dstW: number, dstH: number): GrayImage {
  const out = new Float32Array(dstW * dstH);
  const sx = src.w / dstW;
  const sy = src.h / dstH;
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const fx = (x + 0.5) * sx - 0.5;
      const fy = (y + 0.5) * sy - 0.5;
      const x0 = clamp(Math.floor(fx), 0, src.w - 1);
      const y0 = clamp(Math.floor(fy), 0, src.h - 1);
      const x1 = Math.min(x0 + 1, src.w - 1);
      const y1 = Math.min(y0 + 1, src.h - 1);
      const tx = fx - x0;
      const ty = fy - y0;
      const p00 = src.data[y0 * src.w + x0];
      const p10 = src.data[y0 * src.w + x1];
      const p01 = src.data[y1 * src.w + x0];
      const p11 = src.data[y1 * src.w + x1];
      out[y * dstW + x] =
        (1 - tx) * (1 - ty) * p00 + tx * (1 - ty) * p10 +
        (1 - tx) * ty * p01 + tx * ty * p11;
    }
  }
  return { w: dstW, h: dstH, data: out };
}

export function resizeImageData(src: ImageData, dstW: number, dstH: number): ImageData {
  const canvas = new OffscreenCanvas(dstW, dstH);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  const srcCanvas = new OffscreenCanvas(src.width, src.height);
  const sctx = srcCanvas.getContext('2d')!;
  sctx.putImageData(src, 0, 0);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(srcCanvas, 0, 0, dstW, dstH);
  return ctx.getImageData(0, 0, dstW, dstH);
}

/** 3x3 盒式模糊（可迭代次数） */
export function boxBlur(img: GrayImage, iterations = 1): GrayImage {
  let cur = img;
  for (let it = 0; it < iterations; it++) {
    const out = new Float32Array(cur.w * cur.h);
    const { w, h, data } = cur;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= w) continue;
            s += data[yy * w + xx];
            n++;
          }
        }
        out[y * w + x] = s / n;
      }
    }
    cur = { w, h, data: out };
  }
  return cur;
}

/**
 * 风动掩码（视差鲁棒）：环拍时不同深度竹竿视差不同，纯平移无法对齐。
 * 思路改为“高响应稀疏化”——先取帧差异常点，再只保留邻域响应少的孤立闪烁点
 * （风动叶尖的离散抖动），成片的刚性结构（竹竿竖棱、竹节）不屏蔽。
 * 同时硬性限制掩码占比，保证特征提取始终有足够可分析区域。
 */
export function motionMask(
  prev: GrayImage, cur: GrayImage, threshold = 22, dilateR = 2,
): { mask: Uint8Array; ratio: number } {
  const { w, h } = cur;
  const raw = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (Math.abs(cur.data[i] - prev.data[i]) > threshold) raw[i] = 1;
  }
  // 只保留孤立高响应点，剔除成片的刚性画面结构
  let mask = sparsify(raw, w, h);
  mask = dilate(mask, w, h, dilateR);
  let count = mask.reduce((a, b) => a + b, 0);
  // 占比超过 35% 时判定掩码不可靠，进一步稀疏化，保证特征可用
  if (count / mask.length > 0.35) {
    mask = sparsify(raw, w, h);
    count = mask.reduce((a, b) => a + b, 0);
  }
  return { mask, ratio: count / mask.length };
}

/** 仅保留邻域响应少的孤立点（风动叶尖），成片结构不屏蔽 */
function sparsify(raw: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (!raw[i]) continue;
      let neighbors = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (raw[(y + dy) * w + (x + dx)]) neighbors++;
        }
      }
      if (neighbors <= 3) out[i] = 1;
    }
  }
  return out;
}

export function dilate(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let dy = -r; dy <= r && !v; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          if (mask[yy * w + xx]) { v = 1; break; }
        }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

/** Sobel 梯度 */
export function sobel(img: GrayImage): GradImage {
  const { w, h, data } = img;
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const mag = new Float32Array(w * h);
  const ori = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const tl = data[i - w - 1], t = data[i - w], tr = data[i - w + 1];
      const l = data[i - 1], r = data[i + 1];
      const bl = data[i + w - 1], b = data[i + w], br = data[i + w + 1];
      const sx = tr + 2 * r + br - (tl + 2 * l + bl);
      const sy = bl + 2 * b + br - (tl + 2 * t + tr);
      gx[i] = sx;
      gy[i] = sy;
      mag[i] = Math.hypot(sx, sy);
      ori[i] = (Math.atan2(sy, sx) + Math.PI * 2) % (Math.PI * 2);
    }
  }
  return { w, h, gx, gy, mag, ori };
}

/** 像素 RGB -> HSV（h: 0~360, s/v: 0~1） */
export function rgbToHsv(r: number, g: number, b: number): { h: number; s: number; v: number } {
  const rn = r / 255, gn = g / 255, bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const d = max - min;
  let h = 0;
  if (d > 1e-6) {
    if (max === rn) h = 60 * (((gn - bn) / d) % 6);
    else if (max === gn) h = 60 * ((bn - rn) / d + 2);
    else h = 60 * ((rn - gn) / d + 4);
  }
  if (h < 0) h += 360;
  return { h, s: max === 0 ? 0 : d / max, v: max };
}

/** 取 ImageData 指定区域的平均 RGB（跳过掩码像素） */
export function meanRgb(
  img: ImageData, x0: number, y0: number, x1: number, y1: number, skipMask?: Uint8Array, maskW = 0,
): { r: number; g: number; b: number; n: number } {
  let r = 0, g = 0, b = 0, n = 0;
  const xa = Math.max(0, Math.floor(x0));
  const xb = Math.min(img.width - 1, Math.floor(x1));
  const ya = Math.max(0, Math.floor(y0));
  const yb = Math.min(img.height - 1, Math.floor(y1));
  for (let y = ya; y <= yb; y++) {
    for (let x = xa; x <= xb; x++) {
      if (skipMask && skipMask[y * maskW + x]) continue;
      const i = (y * img.width + x) * 4;
      r += img.data[i]; g += img.data[i + 1]; b += img.data[i + 2];
      n++;
    }
  }
  return n ? { r: r / n, g: g / n, b: b / n, n } : { r: 0, g: 0, b: 0, n: 0 };
}
