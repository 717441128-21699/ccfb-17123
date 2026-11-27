import type { FluteSpec } from './types';

/**
 * 各调门笛子的竹料规格（业内常用经验值，可在界面上编辑）。
 * 直径为竹材胸径外径（毫米），节长要求来自笛身有效管长与膜孔/按孔布局。
 */
export const DEFAULT_FLUTE_SPECS: FluteSpec[] = [
  { key: 'C', label: 'C 调（曲笛，低沉浑厚）', diameterMm: 26, diameterTolMm: 1.5, minNodeSpacingMm: 320, minAge: 3 },
  { key: 'D', label: 'D 调（大曲笛）', diameterMm: 24, diameterTolMm: 1.2, minNodeSpacingMm: 300, minAge: 3 },
  { key: 'E', label: 'E 调（中音笛）', diameterMm: 22, diameterTolMm: 1.0, minNodeSpacingMm: 280, minAge: 3 },
  { key: 'F', label: 'F 调（梆笛）', diameterMm: 20, diameterTolMm: 1.0, minNodeSpacingMm: 250, minAge: 3 },
  { key: 'G', label: 'G 调（小梆笛，明亮高亢）', diameterMm: 18, diameterTolMm: 0.8, minNodeSpacingMm: 220, minAge: 3 },
  { key: 'A', label: 'A 调（小笛）', diameterMm: 16, diameterTolMm: 0.8, minNodeSpacingMm: 200, minAge: 3 },
];

/** 竹龄 -> 竿色（线性 RGB）。新竹带白粉偏青，随竹龄蜡质褪去转黄绿 */
export const AGE_COLORS: Record<number, { r: number; g: number; b: number }> = {
  1: { r: 0.42, g: 0.58, b: 0.36 }, // 青白，白粉重
  2: { r: 0.50, g: 0.62, b: 0.32 },
  3: { r: 0.60, g: 0.66, b: 0.28 },
  4: { r: 0.68, g: 0.66, b: 0.26 }, // 黄绿老竹
  5: { r: 0.72, g: 0.63, b: 0.24 },
};

/** 白粉覆盖比例随竹龄下降 */
export function powderRatio(ageYears: number): number {
  return Math.max(0, 0.55 - 0.11 * ageYears);
}
