// 按笛子调门规格选竹：直径、节长、竹龄、弯曲度综合评分
import type {
  BambooSelection, FluteSpec, ReconstructedCulm,
} from './types';

export function selectCulms(culms: ReconstructedCulm[], specs: FluteSpec[]): BambooSelection[] {
  const selections: BambooSelection[] = [];
  for (const culm of culms) {
    const diameterMm = culm.diameterM * 1000;
    const spacings = nodeSpacingsMm(culm);
    const maxSpacing = spacings.length ? Math.max(...spacings) : 0;
    const bendScore = straightnessScore(culm);

    for (const spec of specs) {
      const reasons: string[] = [];
      const dErr = Math.abs(diameterMm - spec.diameterMm);
      if (dErr > spec.diameterTolMm) continue;
      reasons.push(`胸径 ${diameterMm.toFixed(1)} mm（目标 ${spec.diameterMm}±${spec.diameterTolMm}）`);

      if (maxSpacing < spec.minNodeSpacingMm) continue;
      reasons.push(`最长节间 ${maxSpacing.toFixed(0)} mm（≥${spec.minNodeSpacingMm}）`);

      if (culm.ageYears < spec.minAge) continue;
      reasons.push(`竹龄约 ${culm.ageYears.toFixed(1)} 年（≥${spec.minAge}）`);

      if (bendScore < 0.6) continue;
      reasons.push(`竿直度评分 ${(bendScore * 100).toFixed(0)}`);

      const score = clamp01(
        (1 - dErr / spec.diameterTolMm) * 0.45 +
        Math.min(1, maxSpacing / (spec.minNodeSpacingMm * 1.4)) * 0.3 +
        bendScore * 0.15 +
        culm.ageConfidence * 0.1,
      );
      selections.push({ culmId: culm.id, specKey: spec.key, score, reasons });
    }
  }
  return selections.sort((a, b) => b.score - a.score);
}

export function nodeSpacingsMm(culm: ReconstructedCulm): number[] {
  const out: number[] = [];
  for (let i = 1; i < culm.nodeHeightsM.length; i++) {
    out.push((culm.nodeHeightsM[i] - culm.nodeHeightsM[i - 1]) * 1000);
  }
  return out;
}

/** 竿直度：lean 与 bend 越小越接近 1 */
function straightnessScore(culm: ReconstructedCulm): number {
  const leanMag = Math.hypot(culm.bend.lean.x, culm.bend.lean.y);
  const bendMag = Math.hypot(culm.bend.bend.x, culm.bend.bend.y);
  return clamp01(1 - leanMag * 25 - bendMag * 120);
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}
