// 轻量线性代数（零依赖），全部纯函数，矩阵为行主序 Float32Array(9) / Float64Array(16)
import type { Vec2, Vec3 } from './types';

export const v2 = (x = 0, y = 0): Vec2 => ({ x, y });
export const v3 = (x = 0, y = 0, z = 0): Vec3 => ({ x, y, z });

export const add3 = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const sub3 = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const scale3 = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
export const dot3 = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross3 = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
export const norm3 = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);
export const normalize3 = (a: Vec3): Vec3 => {
  const n = norm3(a);
  return n > 1e-12 ? scale3(a, 1 / n) : v3();
};
export const dist3 = (a: Vec3, b: Vec3): number => norm3(sub3(a, b));

export const add2 = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y });
export const sub2 = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
export const scale2 = (a: Vec2, s: number): Vec2 => ({ x: a.x * s, y: a.y * s });
export const dot2 = (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y;
export const norm2 = (a: Vec2): number => Math.hypot(a.x, a.y);
export const normalize2 = (a: Vec2): Vec2 => {
  const n = norm2(a);
  return n > 1e-12 ? scale2(a, 1 / n) : v2();
};

// ---------- 3x3（行主序，索引 r*3+c） ----------
export type Mat3 = Float32Array;

export const mat3Identity = (): Mat3 => Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);

export function mat3Mul(a: Mat3, b: Mat3): Mat3 {
  const o = new Float32Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      o[r * 3 + c] =
        a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return o;
}

export function mat3Vec(m: Mat3, v: Vec3): Vec3 {
  return {
    x: m[0] * v.x + m[1] * v.y + m[2] * v.z,
    y: m[3] * v.x + m[4] * v.y + m[5] * v.z,
    z: m[6] * v.x + m[7] * v.y + m[8] * v.z,
  };
}

export function mat3Transpose(m: Mat3): Mat3 {
  return Float32Array.of(m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]);
}

/** 由三列（相机坐标系的 x/y/z 轴在世界中的方向）构造旋转矩阵 */
export function mat3FromColumns(c0: Vec3, c1: Vec3, c2: Vec3): Mat3 {
  // 行主序存储，列 c 的元素在 [c, 3+c, 6+c]
  return Float32Array.of(
    c0.x, c1.x, c2.x,
    c0.y, c1.y, c2.y,
    c0.z, c1.z, c2.z,
  );
}

/** look-at：世界 -> 相机的旋转（相机看向 -Z，+Y 朝上） */
export function lookAtRotation(eye: Vec3, target: Vec3, worldUp: Vec3): Mat3 {
  const f = normalize3(sub3(target, eye)); // 相机 -Z 的反方向
  const x = normalize3(cross3(f, worldUp)); // 相机 +X
  const y = cross3(x, f); // 相机 +Y
  // 相机轴在世界中为列；世界->相机 = 轴的转置
  return mat3Transpose(mat3FromColumns(x, y, scale3(f, -1)));
}

// ---------- 4x4（渲染器用，列主序，与 WebGL 一致） ----------
export type Mat4 = Float32Array;

export function mat4Perspective(fovYRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovYRad / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

export function mat4LookAt(eye: Vec3, target: Vec3, worldUp: Vec3): Mat4 {
  const z = normalize3(sub3(eye, target));
  const x = normalize3(cross3(worldUp, z));
  const y = cross3(z, x);
  const m = new Float32Array(16);
  m[0] = x.x; m[4] = x.y; m[8] = x.z; m[12] = -dot3(x, eye);
  m[1] = y.x; m[5] = y.y; m[9] = y.z; m[13] = -dot3(y, eye);
  m[2] = z.x; m[6] = z.y; m[10] = z.z; m[14] = -dot3(z, eye);
  m[15] = 1;
  return m;
}

export function mat4Mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

// ---------- 通用数值 ----------

/** 对 n×m（行主序，n 行 m 列）矩阵做 SVD，返回 { U(n×n), S(min), Vt(m×m) }。
 *  采用 Jacobi 单边正交化（对 A^T A 求特征向量），规模小、够用且无依赖。 */
export function svd(A: Float64Array, n: number, m: number): {
  U: Float64Array; S: Float64Array; Vt: Float64Array;
} {
  // V 通过对 A^T A 做 Jacobi 特征分解
  const AtA = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += A[k * m + i] * A[k * m + j];
      AtA[i * m + j] = s;
    }
  }
  const V = eye64(m);
  jacobiEigen(AtA, V, m);
  // AtA 被覆盖为特征值对角；提取奇异值
  const Sraw = new Float64Array(m);
  for (let i = 0; i < m; i++) Sraw[i] = Math.sqrt(Math.max(0, AtA[i * m + i]));
  // Jacobi 不保证特征值有序，按奇异值降序排列 V 的列
  const order = Array.from({ length: m }, (_, i) => i)
    .sort((a, b) => Sraw[b] - Sraw[a]);
  const Vsorted = new Float64Array(m * m);
  for (let j = 0; j < m; j++) {
    const src = order[j];
    for (let i = 0; i < m; i++) Vsorted[i * m + j] = V[i * m + src];
  }
  const S = new Float64Array(m);
  for (let j = 0; j < m; j++) S[j] = Sraw[order[j]];
  // U = A V S^-1
  const U = new Float64Array(n * n);
  const Ucols: Float64Array[] = [];
  for (let j = 0; j < m; j++) {
    const col = new Float64Array(n);
    const inv = S[j] > 1e-12 ? 1 / S[j] : 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += A[i * m + k] * Vsorted[k * m + j];
      col[i] = s * inv;
    }
    Ucols.push(col);
  }
  // n==m 时直接填列；n>m（如 8 点法 n>>m）由调用方自行处理，这里只支持方阵/瘦矩阵
  for (let j = 0; j < Math.min(n, m); j++) {
    for (let i = 0; i < n; i++) U[i * n + j] = Ucols[j][i];
  }
  if (n === m) {
    // 补全可能缺失的正交列（零奇异值情形）
    fillOrthonormal(U, n);
  }
  const Vt = transpose64(Vsorted, m, m);
  return { U, S, Vt };
}

function eye64(n: number): Float64Array {
  const m = new Float64Array(n * n);
  for (let i = 0; i < n; i++) m[i * n + i] = 1;
  return m;
}

function transpose64(a: Float64Array, n: number, m: number): Float64Array {
  const o = new Float64Array(m * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) o[j * n + i] = a[i * m + j];
  return o;
}

function jacobiEigen(A: Float64Array, V: Float64Array, n: number, maxIter = 60): void {
  for (let iter = 0; iter < maxIter; iter++) {
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-18) continue;
        const app = A[p * n + p];
        const aqq = A[q * n + q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p];
          const akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k];
          const aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p];
          const vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
}

/** 对列正交矩阵 U 补全零空间列（Gram-Schmidt） */
function fillOrthonormal(U: Float64Array, n: number): void {
  for (let col = 0; col < n; col++) {
    const v = new Float64Array(n);
    let norm = 0;
    for (let i = 0; i < n; i++) {
      norm += U[i * n + col] ** 2;
    }
    if (norm > 1e-20) continue;
    // 找一个与已有列正交的向量
    for (let cand = 0; cand < n; cand++) {
      for (let i = 0; i < n; i++) v[i] = i === cand ? 1 : 0;
      for (let j = 0; j < col; j++) {
        let d = 0;
        for (let i = 0; i < n; i++) d += U[i * n + j] * v[i];
        for (let i = 0; i < n; i++) v[i] -= d * U[i * n + j];
      }
      const nn = Math.hypot(...v);
      if (nn > 1e-9) {
        for (let i = 0; i < n; i++) U[i * n + col] = v[i] / nn;
        break;
      }
    }
  }
}

/** 解标准最小二乘 min ||J x - r||，J 为行主序 (k 行 p 列)，正规方程法（k 通常远大于 p）。
 *  Gauss-Newton 场景下残差为 f 时传入 -f 作为右端项即可得到更新量 δ。 */
export function solveLeastSquares(J: Float64Array, r: Float64Array, k: number, p: number): Float64Array {
  const H = new Float64Array(p * p);
  const g = new Float64Array(p);
  for (let i = 0; i < k; i++) {
    for (let a = 0; a < p; a++) {
      g[a] += J[i * p + a] * r[i];
      for (let b = 0; b < p; b++) {
        H[a * p + b] += J[i * p + a] * J[i * p + b];
      }
    }
  }
  // Levenberg-Marquardt 阻尼（相对对角尺度，避免良态问题被过度正则）
  const damp = 1e-9 * Math.max(1, traceOf(H, p));
  for (let a = 0; a < p; a++) H[a * p + a] += damp;
  return solveSymmetric(H, g, p);
}

function traceOf(H: Float64Array, p: number): number {
  let t = 0;
  for (let i = 0; i < p; i++) t += H[i * p + i];
  return t / p;
}

/** Cholesky 解对称正定方程组 H x = b */
export function solveSymmetric(H: Float64Array, b: Float64Array, p: number): Float64Array {
  const L = new Float64Array(p * p);
  for (let j = 0; j < p; j++) {
    for (let i = j; i < p; i++) {
      let s = H[i * p + j];
      for (let k = 0; k < j; k++) s -= L[i * p + k] * L[j * p + k];
      if (i === j) {
        if (s <= 1e-18) s = 1e-18; // 退化保护
        L[i * p + j] = Math.sqrt(s);
      } else {
        L[i * p + j] = s / L[j * p + j];
      }
    }
  }
  const y = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * p + k] * y[k];
    y[i] = s / L[i * p + i];
  }
  const x = new Float64Array(p);
  for (let i = p - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < p; k++) s -= L[k * p + i] * x[k];
    x[i] = s / L[i * p + i];
  }
  return x;
}

/** 随机打乱（Fisher-Yates，使用注入的随机函数，保证可复现） */
export function shuffle<T>(arr: T[], rand: () => number): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
export const deg2rad = (d: number): number => (d * Math.PI) / 180;
export const rad2deg = (r: number): number => (r * 180) / Math.PI;
