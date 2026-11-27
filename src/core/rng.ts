// 可复现随机数：mulberry32 + 独立流的高斯采样

export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** [0, 1) 均匀分布 */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** [lo, hi) */
  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /** 正态分布（Box-Muller） */
  gaussian(mean = 0, std = 1): number {
    const u = Math.max(1e-12, this.next());
    const v = this.next();
    return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** 派生独立子流（同种子可复现） */
  stream(salt: number): Rng {
    let h = this.state ^ Math.imul(salt, 0x9e3779b9);
    h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
    h ^= h >>> 13;
    return new Rng(h >>> 0);
  }
}
