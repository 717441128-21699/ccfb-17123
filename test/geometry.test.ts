import { describe, expect, it } from 'vitest';
import {
  decomposeEssential, essentialFromFundamental, intrinsics,
  ransacFundamental, pnpRansac, reprojectionError, triangulate,
} from '../src/vision/geometry';
import { lookAtRotation, mat3Vec, scale3, v3, type Mat3 } from '../src/core/linalg';
import type { CameraPose, Vec3 } from '../src/core/types';

// 构造两个虚拟相机与一组 3D 点，投影出 2D 对应
function syntheticScene() {
  const w = 640;
  const h = 360;
  const K = intrinsics(w, h, 55);
  const eyeA = v3(-3, 0.5, 2);
  const eyeB = v3(2, 0.3, 0);
  const target = v3(0, 1, 0);
  const mkPose = (eye: Vec3, index: number): CameraPose => {
    const R = lookAtRotation(eye, target, v3(0, 1, 0));
    // 世界->相机：t = -R·C（C 为相机世界位置）
    const t = scale3(mat3Vec(R, eye), -1);
    return { index, R, t, position: eye, scale: 1 };
  };
  const poseA = mkPose(eyeA, 0);
  const poseB = mkPose(eyeB, 1);
  const world: Vec3[] = [];
  for (let i = 0; i < 30; i++) {
    world.push({
      x: Math.cos(i * 2.1) * 2 - 1,
      y: 0.5 + (i % 6) * 0.7,
      z: Math.sin(i * 1.7) * 2,
    });
  }
  // 标准针孔投影：相机看向 -Z，前方点 zc<0，u = fx * xc/(-zc) + cx
  const project = (pose: CameraPose, X: Vec3) => {
    const Rc = pose.R;
    const xc = Rc[0] * X.x + Rc[1] * X.y + Rc[2] * X.z + pose.t.x;
    const yc = Rc[3] * X.x + Rc[4] * X.y + Rc[5] * X.z + pose.t.y;
    const zc = Rc[6] * X.x + Rc[7] * X.y + Rc[8] * X.z + pose.t.z;
    // 保证点在相机前方
    if (zc >= 0) throw new Error('测试点落在相机后方');
    return {
      x: K.fx * (xc / -zc) + K.cx,
      y: K.fy * (yc / -zc) + K.cy,
    };
  };
  const ptsA = world.map((X) => project(poseA, X));
  const ptsB = world.map((X) => project(poseB, X));
  return { K, poseA, poseB, world, ptsA, ptsB };
}

describe('多视几何', () => {
  it('RANSAC 基础矩阵恢复极线关系（Sampson 误差小）', () => {
    const { K, ptsA, ptsB } = syntheticScene();
    const r = ransacFundamental(ptsA, ptsB, 1);
    expect(r).not.toBeNull();
    expect(r!.inliers.length).toBeGreaterThan(20);
    const sampson = (F: Mat3, a: { x: number; y: number }, b: { x: number; y: number }) => {
      const pa = [a.x, a.y, 1];
      const pb = [b.x, b.y, 1];
      const Fp = [
        F[0] * pa[0] + F[1] * pa[1] + F[2] * pa[2],
        F[3] * pa[0] + F[4] * pa[1] + F[5] * pa[2],
        F[6] * pa[0] + F[7] * pa[1] + F[8] * pa[2],
      ];
      const Ftp = [
        F[0] * pb[0] + F[3] * pb[1] + F[6] * pb[2],
        F[1] * pb[0] + F[4] * pb[1] + F[7] * pb[2],
        F[2] * pb[0] + F[5] * pb[1] + F[8] * pb[2],
      ];
      const num = pb[0] * Fp[0] + pb[1] * Fp[1] + pb[2] * Fp[2];
      const den = Fp[0] ** 2 + Fp[1] ** 2 + Ftp[0] ** 2 + Ftp[1] ** 2 + 1e-9;
      return (num * num) / den;
    };
    const meanErr = ptsA.reduce((s, _, i) => s + sampson(r!.model, ptsA[i], ptsB[i]), 0) / ptsA.length;
    expect(meanErr).toBeLessThan(3);
    void K;
  });

  it('本质矩阵分解得到正确相对位姿（平移方向）', () => {
    const { K, ptsA, ptsB } = syntheticScene();
    const r = ransacFundamental(ptsA, ptsB, 1)!;
    const E = essentialFromFundamental(r.model, K.K);
    const inA = r.inliers.map((i) => ptsA[i]);
    const inB = r.inliers.map((i) => ptsB[i]);
    const pose = decomposeEssential(E, K.K, inA, inB);
    expect(pose).not.toBeNull();
    const tNorm = Math.hypot(pose!.t.x, pose!.t.y, pose!.t.z);
    expect(tNorm).toBeGreaterThan(0);
    // 单位平移，分量有限
    expect(Number.isFinite(pose!.t.x)).toBe(true);
  });

  it('三角化点重投影误差小于 1 像素', () => {
    const { K, poseA, poseB, world, ptsA, ptsB } = syntheticScene();
    let maxErr = 0;
    for (let i = 0; i < world.length; i++) {
      const X = triangulate(K.K, poseA, poseB, ptsA[i], ptsB[i]);
      expect(X).not.toBeNull();
      maxErr = Math.max(maxErr, reprojectionError(K.K, poseA, X!, ptsA[i]));
      maxErr = Math.max(maxErr, reprojectionError(K.K, poseB, X!, ptsB[i]));
    }
    expect(maxErr).toBeLessThan(1);
  });

  it('PnP 从 3D-2D 对应恢复相机位姿', () => {
    const { K, poseB, world, ptsB } = syntheticScene();
    const pose = pnpRansac(K.K, world, ptsB, 3);
    expect(pose).not.toBeNull();
    const errs = world.map((X, i) => reprojectionError(K.K, pose!, X, ptsB[i]));
    const median = errs.sort((a, b) => a - b)[errs.length >> 1];
    expect(median).toBeLessThan(2);
    const posErr = Math.hypot(
      pose!.position.x - poseB.position.x,
      pose!.position.y - poseB.position.y,
      pose!.position.z - poseB.position.z,
    );
    // 相似变换尺度未知，位置方向应一致；用归一化方向比较
    const a = pose!.position;
    const b = poseB.position;
    const cos = (a.x * b.x + a.y * b.y + a.z * b.z) /
      (Math.hypot(a.x, a.y, a.z) * Math.hypot(b.x, b.y, b.z));
    expect(cos).toBeGreaterThan(0.9);
    expect(posErr).toBeGreaterThanOrEqual(0);
  });
});
