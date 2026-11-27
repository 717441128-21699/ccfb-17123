import { describe, expect, it } from 'vitest';
import {
  lookAtRotation, mat3Mul, mat3Transpose, mat3Vec, normalize3, solveLeastSquares, svd, v3,
} from '../src/core/linalg';

describe('linalg', () => {
  it('lookAt 旋转把视线方向映射到相机 -Z', () => {
    const eye = v3(5, 1.5, 0);
    const target = v3(0, 1.5, 0);
    const R = lookAtRotation(eye, target, v3(0, 1, 0));
    const fwd = normalize3(v3(target.x - eye.x, 0, target.z - eye.z));
    const cam = mat3Vec(R, fwd);
    expect(cam.z).toBeLessThan(-0.999);
    expect(Math.abs(cam.x)).toBeLessThan(1e-6);
    expect(Math.abs(cam.y)).toBeLessThan(1e-6);
  });

  it('旋转矩阵正交：R^T R = I', () => {
    const R = lookAtRotation(v3(3, 2, 4), v3(0, 1, 0), v3(0, 1, 0));
    const RtR = mat3Mul(mat3Transpose(R), R);
    for (let i = 0; i < 9; i++) {
      const expected = i % 4 === 0 ? 1 : 0;
      expect(RtR[i]).toBeCloseTo(expected, 6);
    }
  });

  it('SVD 重构 A = U S Vt', () => {
    const A = Float64Array.of(1, 2, 3, 4, 5, 6, 7, 8, 10);
    const { U, S, Vt } = svd(A, 3, 3);
    const Sm = Float64Array.of(S[0], 0, 0, 0, S[1], 0, 0, 0, S[2]);
    const US = mul(U, Sm);
    const recon = mul(US, Vt);
    for (let i = 0; i < 9; i++) expect(recon[i]).toBeCloseTo(A[i], 5);
    // 奇异值降序
    expect(S[0]).toBeGreaterThanOrEqual(S[1]);
    expect(S[1]).toBeGreaterThanOrEqual(S[2]);
  });

  it('最小二乘解线性回归 y = 2x + 3', () => {
    // 模型 y = a*x + b，6 个样本
    const xs = [0, 1, 2, 3, 4, 5];
    const ys = xs.map((x) => 2 * x + 3);
    const J = new Float64Array(xs.length * 2);
    const r = new Float64Array(xs.length);
    xs.forEach((x, i) => {
      J[i * 2] = x;
      J[i * 2 + 1] = 1;
      r[i] = ys[i];
    });
    // solveLeastSquares 解的是 J x = r（这里直接求参数）
    const params = solveLeastSquares(J, r, xs.length, 2);
    expect(params[0]).toBeCloseTo(2, 4);
    expect(params[1]).toBeCloseTo(3, 4);
  });
});

function mul(a: Float64Array, b: Float64Array): Float64Array {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) o[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return o;
}
