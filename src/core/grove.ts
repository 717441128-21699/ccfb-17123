// 合成竹林生成：在圆形样地内按泊松盘排斥布竿，参数按竹龄/密度分布，全部可复现
import { Rng } from './rng';
import { AGE_COLORS, powderRatio } from './specs';
import type {
  BendModel, GroundTruthCulm, GroveGroundTruth, GroveGenParams,
  ReferencePole, Vec2, Vec3,
} from './types';

export const DENSITY_PRESETS: Record<string, { count: number; radiusM: number }> = {
  sparse: { count: 48, radiusM: 11 },
  medium: { count: 96, radiusM: 11 },
  dense: { count: 160, radiusM: 11 },
};

export const DEFAULT_GEN_PARAMS: Omit<GroveGenParams, 'density' | 'wind' | 'seed' | 'culmCount' | 'plotRadiusM'> = {
  frameCount: 300,
  width: 1280,
  height: 720,
  fovYDeg: 55,
  orbitRadiusM: 14,
};

export function defaultGenParams(
  density: GroveGenParams['density'],
  wind: GroveGenParams['wind'],
  seed: number,
): GroveGenParams {
  const preset = DENSITY_PRESETS[density];
  return {
    density, wind, seed,
    culmCount: preset.count,
    plotRadiusM: preset.radiusM,
    ...DEFAULT_GEN_PARAMS,
  };
}

/** 泊松盘布点：最小间距随密度变化，保证相邻竹竿不会密到无法区分 */
function poissonDisc(rng: Rng, count: number, radius: number, minDist: number): Vec2[] {
  const pts: Vec2[] = [];
  let attempts = 0;
  const maxAttempts = count * 200;
  while (pts.length < count && attempts < maxAttempts) {
    attempts++;
    const ang = rng.range(0, Math.PI * 2);
    const r = radius * Math.sqrt(rng.next());
    const p = { x: r * Math.cos(ang), y: r * Math.sin(ang) };
    let ok = true;
    for (const q of pts) {
      if (Math.hypot(p.x - q.x, p.y - q.y) < minDist) { ok = false; break; }
    }
    if (ok) pts.push(p);
  }
  return pts;
}

export function generateGrove(params: GroveGenParams): GroveGroundTruth {
  const rng = new Rng(params.seed);
  const minDist = params.density === 'dense' ? 0.55 : params.density === 'medium' ? 0.75 : 1.05;
  const positions = poissonDisc(rng, params.culmCount, params.plotRadiusM * 0.92, minDist);

  const culms: GroundTruthCulm[] = positions.map((pos, i) => {
    const ageYears = 1 + Math.floor(rng.next() * 5); // 1..5
    // 胸径 15~90 mm，老竹略粗
    const diameterM = (0.015 + rng.next() * 0.06) * (0.85 + 0.06 * ageYears);
    const heightM = rng.range(7, 14);
    // 竹节：基部密、上部疏，节间距 180~460 mm
    const nodeHeightsM: number[] = [];
    let z = rng.range(0.08, 0.2);
    while (z < heightM * 0.94) {
      nodeHeightsM.push(z);
      const spacing = rng.range(0.2, 0.34) * (0.8 + 0.5 * (z / heightM));
      z += spacing;
    }
    const bend: BendModel = {
      base: pos,
      lean: { x: rng.gaussian(0, 0.012), y: rng.gaussian(0, 0.012) },
      bend: { x: rng.gaussian(0, 0.0025), y: rng.gaussian(0, 0.0025) },
    };
    const baseColor = ageColorJitter(rng, ageYears);
    // 竿面斑痕 3~10 处（跨帧匹配的高区分度特征）
    const spots = Array.from({ length: 3 + Math.floor(rng.next() * 8) }, () => ({
      u: rng.range(0.08, 0.9),
      theta: rng.range(0, Math.PI * 2),
      r: rng.range(0.008, 0.022),
      darkness: rng.range(0.25, 0.6),
    }));
    return { id: i, bend, heightM, diameterM, nodeHeightsM, ageYears, baseColor, spots };
  });

  // 标尺竿：红漆、已知直径，放在环拍中心附近、视野开阔处
  const pole: ReferencePole = {
    bend: {
      base: { x: rng.range(-1.5, 1.5), y: rng.range(-1.5, 1.5) },
      lean: { x: 0, y: 0 },
      bend: { x: 0, y: 0 },
    },
    heightM: 2.5,
    diameterM: 0.03,
  };

  // 环拍相机：绕样地一圈，轻微高低起伏，始终注视中心
  const frames = Array.from({ length: params.frameCount }, (_, i) => {
    const t = i / params.frameCount;
    const ang = t * Math.PI * 2;
    const wobble = 0.35 * Math.sin(ang * 3 + 0.7); // 手持起伏
    const position: Vec3 = {
      x: Math.cos(ang) * params.orbitRadiusM,
      y: 1.55 + 0.12 * Math.sin(ang * 2) + wobble * 0.2,
      z: Math.sin(ang) * params.orbitRadiusM,
    };
    const target: Vec3 = { x: 0, y: 1.7, z: 0 };
    return { index: i, position, target, up: { x: 0, y: 1, z: 0 }, fovYDeg: params.fovYDeg };
  });

  return { culms, pole, frames, params };
}

function ageColorJitter(rng: Rng, ageYears: number): Vec3 {
  const c = AGE_COLORS[ageYears];
  return {
    x: Math.min(1, c.r + rng.gaussian(0, 0.025)),
    y: Math.min(1, c.g + rng.gaussian(0, 0.025)),
    z: Math.min(1, c.b + rng.gaussian(0, 0.02)),
  };
}

/** 弯曲模型在高度 z 处的水平位置 */
export function bendAxisAt(bend: BendModel, z: number): Vec2 {
  return {
    x: bend.base.x + bend.lean.x * z + bend.bend.x * z * z,
    y: bend.base.y + bend.lean.y * z + bend.bend.y * z * z,
  };
}

export { powderRatio };
