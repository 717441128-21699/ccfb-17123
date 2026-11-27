import { describe, expect, it } from 'vitest';
import { generateGrove, bendAxisAt, defaultGenParams } from '../src/core/grove';
import { Rng } from '../src/core/rng';

describe('grove generation', () => {
  const params = defaultGenParams('medium', 'breeze', 123);

  it('同种子可复现', () => {
    const a = generateGrove(params);
    const b = generateGrove(params);
    expect(a.culms.length).toBe(b.culms.length);
    expect(a.culms[0].bend.base.x).toBe(b.culms[0].bend.base.x);
    expect(a.culms[5].diameterM).toBe(b.culms[5].diameterM);
  });

  it('竹竿数量与密度预设一致', () => {
    const g = generateGrove(params);
    expect(g.culms.length).toBe(96);
    expect(g.frames.length).toBe(params.frameCount);
  });

  it('泊松盘最小间距保证竹竿不重叠', () => {
    const g = generateGrove(defaultGenParams('dense', 'calm', 7));
    let minD = Infinity;
    for (let i = 0; i < g.culms.length; i++) {
      for (let j = i + 1; j < g.culms.length; j++) {
        const d = Math.hypot(
          g.culms[i].bend.base.x - g.culms[j].bend.base.x,
          g.culms[i].bend.base.y - g.culms[j].bend.base.y,
        );
        minD = Math.min(minD, d);
      }
    }
    expect(minD).toBeGreaterThan(0.5);
  });

  it('竹节升序且在竿高范围内', () => {
    const g = generateGrove(params);
    for (const c of g.culms) {
      expect(c.nodeHeightsM.length).toBeGreaterThan(10);
      for (let i = 1; i < c.nodeHeightsM.length; i++) {
        expect(c.nodeHeightsM[i]).toBeGreaterThan(c.nodeHeightsM[i - 1]);
      }
      expect(c.nodeHeightsM[c.nodeHeightsM.length - 1]).toBeLessThan(c.heightM);
    }
  });

  it('弯曲轴线随高度单调偏移', () => {
    const g = generateGrove(params);
    const c = g.culms[0];
    const p0 = bendAxisAt(c.bend, 0);
    const p5 = bendAxisAt(c.bend, 5);
    expect(p0.x).toBe(c.bend.base.x);
    // 偏移量有限（无飞值）
    expect(Math.abs(p5.x)).toBeLessThan(20);
  });

  it('竹龄在 1~5 之间且直径为正', () => {
    const g = generateGrove(params);
    for (const c of g.culms) {
      expect(c.ageYears).toBeGreaterThanOrEqual(1);
      expect(c.ageYears).toBeLessThanOrEqual(5);
      expect(c.diameterM).toBeGreaterThan(0.01);
      expect(c.diameterM).toBeLessThan(0.12);
    }
  });
});

describe('Rng', () => {
  it('同种子序列一致', () => {
    const a = new Rng(42);
    const b = new Rng(42);
    for (let i = 0; i < 10; i++) expect(a.next()).toBe(b.next());
  });

  it('高斯采样均值与标准差近似正确', () => {
    const rng = new Rng(99);
    const vals = Array.from({ length: 20000 }, () => rng.gaussian(5, 2));
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
    expect(mean).toBeCloseTo(5, 1);
    expect(std).toBeCloseTo(2, 1);
  });
});
