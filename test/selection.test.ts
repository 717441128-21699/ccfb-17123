import { describe, expect, it } from 'vitest';
import { selectCulms, nodeSpacingsMm } from '../src/core/selection';
import { evaluate } from '../src/core/evaluate';
import { DEFAULT_FLUTE_SPECS } from '../src/core/specs';
import type { GroundTruthCulm, ReconstructedCulm } from '../src/core/types';

function makeRecon(over: Partial<ReconstructedCulm>): ReconstructedCulm {
  return {
    id: 0,
    bend: { base: { x: 0, y: 0 }, lean: { x: 0, y: 0 }, bend: { x: 0, y: 0 } },
    diameterM: 0.02,
    diameterSigmaM: 0.0005,
    nodeHeightsM: [0.2, 0.55, 0.92, 1.3, 1.72],
    ageYears: 4,
    ageConfidence: 0.8,
    meanColor: { x: 0.6, y: 0.6, z: 0.3 },
    observationCount: 12,
    reprojectionErrorPx: 1.2,
    ...over,
  };
}

function makeTruth(over: Partial<GroundTruthCulm>): GroundTruthCulm {
  return {
    id: 0,
    bend: { base: { x: 0, y: 0 }, lean: { x: 0, y: 0 }, bend: { x: 0, y: 0 } },
    heightM: 10,
    diameterM: 0.02,
    nodeHeightsM: [0.2, 0.55, 0.92, 1.3, 1.72],
    ageYears: 4,
    baseColor: { x: 0.6, y: 0.6, z: 0.3 },
    spots: [],
    ...over,
  };
}

describe('选竹', () => {
  it('直径/节长/竹龄合格的竹竿被选中并给出评分', () => {
    // G 调目标 18mm，容差 0.8；造一根 18.2mm、长节、4 年竹
    const culm = makeRecon({ diameterM: 0.0182 });
    const sel = selectCulms([culm], DEFAULT_FLUTE_SPECS);
    expect(sel.length).toBeGreaterThan(0);
    expect(sel.some((s) => s.specKey === 'G')).toBe(true);
    expect(sel[0].score).toBeGreaterThan(0.5);
  });

  it('直径超出容差的竹竿不入选任何调门', () => {
    const culm = makeRecon({ diameterM: 0.09 }); // 90mm，无对应调门
    const sel = selectCulms([culm], DEFAULT_FLUTE_SPECS);
    expect(sel.length).toBe(0);
  });

  it('新竹（竹龄不足）被排除', () => {
    const culm = makeRecon({ diameterM: 0.018, ageYears: 1 });
    const sel = selectCulms([culm], DEFAULT_FLUTE_SPECS);
    expect(sel.length).toBe(0);
  });

  it('节间距计算正确（毫米）', () => {
    const culm = makeRecon({});
    const sp = nodeSpacingsMm(culm);
    expect(sp[0]).toBeCloseTo(350, 5);
    expect(sp.length).toBe(4);
  });
});

describe('评测', () => {
  it('完全一致的重建误差为 0', () => {
    const truth = [makeTruth({ id: 0 }), makeTruth({ id: 1, bend: { base: { x: 2, y: 1 }, lean: { x: 0, y: 0 }, bend: { x: 0, y: 0 } } })];
    const recon = truth.map((t) => makeRecon({
      id: t.id,
      bend: t.bend,
      diameterM: t.diameterM,
      nodeHeightsM: t.nodeHeightsM,
      ageYears: t.ageYears,
    }));
    const rep = evaluate(truth, recon);
    expect(rep.positionRmseM).toBeLessThan(1e-9);
    expect(rep.diameterRmseMm).toBeLessThan(1e-9);
    expect(rep.nodeSpacingRmseMm).toBeLessThan(1e-9);
    expect(rep.splitErrors).toBe(0);
    expect(rep.mergeErrors).toBe(0);
    expect(rep.ageAccuracy).toBe(1);
  });

  it('位置偏差 5cm 被正确测出（交替方向位移，旋转/尺度对齐均无法吸收）', () => {
    const truth = Array.from({ length: 8 }, (_, i) => {
      const ang = (i / 8) * Math.PI * 2;
      return makeTruth({
        id: i,
        bend: {
          base: { x: Math.cos(ang) * 4, y: Math.sin(ang) * 4 },
          lean: { x: 0, y: 0 }, bend: { x: 0, y: 0 },
        },
      });
    });
    const recon = truth.map((t, i) => {
      // 相邻点沿相反的切向偏移，尺度/旋转对齐无法消除
      const sign = i % 2 === 0 ? 1 : -1;
      const n = Math.hypot(t.bend.base.x, t.bend.base.y) || 1;
      const tx = -t.bend.base.y / n;
      const ty = t.bend.base.x / n;
      return makeRecon({
        id: t.id,
        bend: {
          ...t.bend,
          base: {
            x: t.bend.base.x + tx * sign * 0.05,
            y: t.bend.base.y + ty * sign * 0.05,
          },
        },
      });
    });
    const rep = evaluate(truth, recon);
    expect(rep.positionRmseM).toBeCloseTo(0.05, 2);
  });

  it('胸径偏差 1mm 被测出', () => {
    const truth = [makeTruth({ id: 0, diameterM: 0.02 })];
    const recon = [makeRecon({ id: 0, diameterM: 0.021 })];
    const rep = evaluate(truth, recon);
    expect(rep.diameterRmseMm).toBeCloseTo(1, 4);
  });

  it('一根真值对应两根重建 -> 拆棒错误', () => {
    const truth = [makeTruth({ id: 0 })];
    const recon = [
      makeRecon({ id: 0 }),
      makeRecon({ id: 1, bend: { base: { x: 0.1, y: 0 }, lean: { x: 0, y: 0 }, bend: { x: 0, y: 0 } } }),
    ];
    const rep = evaluate(truth, recon);
    expect(rep.splitErrors).toBeGreaterThan(0);
  });
});
