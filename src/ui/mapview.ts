// Canvas 2D 俯视图：竹竿位置、编号、调门标注、相机轨迹
import type {
  GroundTruthCulm, ReconstructedCulm, CameraPose, BambooSelection,
} from '../core/types';
import { bendAxisAt } from '../core/grove';

export interface MapViewOptions {
  showTruth: boolean;
  showRecon: boolean;
  showTrajectory: boolean;
  showLabels: boolean;
  highlightSpec: string | null;
  selections: BambooSelection[];
  selectedCulmId: number | null;
  onSelect?: (id: number | null) => void;
}

export class TopViewMap {
  private scale = 14;
  private offsetX = 0;
  private offsetY = 0;
  private hitAreas: { id: number; x: number; y: number; r: number }[] = [];

  constructor(private canvas: HTMLCanvasElement) {
    canvas.addEventListener('click', (e) => this.handleClick(e));
  }

  private handleClick(e: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (this.canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (this.canvas.height / rect.height);
    let hit: number | null = null;
    for (const h of this.hitAreas) {
      if (Math.hypot(h.x - mx, h.y - my) <= h.r + 3) {
        hit = h.id;
        break;
      }
    }
    this.lastOptions?.onSelect?.(hit);
  }

  private lastOptions: MapViewOptions | null = null;

  render(
    truth: GroundTruthCulm[] | null,
    recon: ReconstructedCulm[],
    poses: CameraPose[],
    opts: MapViewOptions,
  ): void {
    this.lastOptions = opts;
    const ctx = this.canvas.getContext('2d')!;
    const W = this.canvas.width;
    const H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);
    // 自动取景
    const all: { x: number; y: number }[] = [];
    recon.forEach((c) => all.push(c.bend.base));
    truth?.forEach((c) => all.push(c.bend.base));
    poses.forEach((p) => all.push({ x: p.position.x, y: p.position.z }));
    if (!all.length) {
      ctx.fillStyle = '#8a9583';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('暂无重建数据', W / 2, H / 2);
      return;
    }
    const minX = Math.min(...all.map((p) => p.x));
    const maxX = Math.max(...all.map((p) => p.x));
    const minY = Math.min(...all.map((p) => p.y));
    const maxY = Math.max(...all.map((p) => p.y));
    const span = Math.max(2, Math.max(maxX - minX, maxY - minY));
    this.scale = Math.min(W, H) * 0.82 / span;
    this.offsetX = (minX + maxX) / 2;
    this.offsetY = (minY + maxY) / 2;
    const tx = (x: number) => W / 2 + (x - this.offsetX) * this.scale;
    const ty = (y: number) => H / 2 + (y - this.offsetY) * this.scale;

    // 网格与指北针
    this.drawGrid(ctx, W, H, tx, ty);
    this.drawCompass(ctx, W);

    // 相机轨迹
    if (opts.showTrajectory && poses.length) {
      ctx.strokeStyle = '#2c4d70';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      poses.forEach((p, i) => {
        const x = tx(p.position.x);
        const y = ty(p.position.z);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      poses.forEach((p, i) => {
        if (i % Math.max(1, Math.floor(poses.length / 12)) !== 0) return;
        ctx.fillStyle = '#2c4d70';
        ctx.beginPath();
        ctx.arc(tx(p.position.x), ty(p.position.z), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // 真值（空心圆）
    if (opts.showTruth && truth) {
      truth.forEach((c) => {
        ctx.strokeStyle = 'rgba(110,120,105,0.7)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(tx(c.bend.base.x), ty(c.bend.base.y), 4, 0, Math.PI * 2);
        ctx.stroke();
      });
    }

    // 选竹结果索引
    const selByCulm = new Map<number, BambooSelection>();
    opts.selections.forEach((s) => {
      if (!selByCulm.has(s.culmId) || s.score > selByCulm.get(s.culmId)!.score) {
        selByCulm.set(s.culmId, s);
      }
    });

    // 重建竹竿
    this.hitAreas = [];
    recon.forEach((c) => {
      const x = tx(c.bend.base.x);
      const y = ty(c.bend.base.y);
      const r = Math.max(4, c.diameterM * this.scale * 18);
      const sel = selByCulm.get(c.id);
      const isHighlight = opts.highlightSpec && sel?.specKey === opts.highlightSpec;
      const isSelected = opts.selectedCulmId === c.id;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      if (isHighlight) {
        ctx.fillStyle = specColor(sel!.specKey);
        ctx.fill();
        ctx.strokeStyle = '#1f2a1d';
        ctx.lineWidth = 2;
        ctx.stroke();
      } else {
        ctx.fillStyle = ageColor(c.ageYears);
        ctx.globalAlpha = 0.85;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = 'rgba(31,42,29,0.5)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      if (isSelected) {
        ctx.strokeStyle = '#c28a2e';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, r + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
      this.hitAreas.push({ id: c.id, x, y, r });
      if (opts.showLabels && (isHighlight || isSelected || recon.length < 60)) {
        ctx.fillStyle = '#1f2a1d';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(String(c.id), x, y - r - 3);
      }
    });
  }

  private drawGrid(
    ctx: CanvasRenderingContext2D, W: number, H: number,
    tx: (x: number) => number, ty: (y: number) => number,
  ): void {
    ctx.strokeStyle = 'rgba(120,135,110,0.18)';
    ctx.lineWidth = 1;
    for (let g = -20; g <= 20; g += 2) {
      ctx.beginPath();
      ctx.moveTo(tx(g), 0);
      ctx.lineTo(tx(g), H);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, ty(g));
      ctx.lineTo(W, ty(g));
      ctx.stroke();
    }
  }

  private drawCompass(ctx: CanvasRenderingContext2D, W: number): void {
    const x = W - 26;
    const y = 28;
    ctx.fillStyle = '#2e3a2b';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('北', x, y - 12);
    ctx.beginPath();
    ctx.moveTo(x, y - 8);
    ctx.lineTo(x - 5, y + 6);
    ctx.lineTo(x, y + 2);
    ctx.lineTo(x + 5, y + 6);
    ctx.closePath();
    ctx.fillStyle = '#b0432f';
    ctx.fill();
  }
}

export function specColor(key: string): string {
  const map: Record<string, string> = {
    C: '#d94f3a', D: '#e08a2e', E: '#d9c22e',
    F: '#5aa84a', G: '#2f7d8f', A: '#6a4f9e',
  };
  return map[key] ?? '#555';
}

export function ageColor(ageYears: number): string {
  // 新竹青绿 -> 老竹黄绿
  const t = Math.max(0, Math.min(1, (ageYears - 1) / 4));
  const r = Math.round(90 + t * 110);
  const g = Math.round(150 - t * 10);
  const b = Math.round(90 - t * 35);
  return `rgb(${r},${g},${b})`;
}

export { bendAxisAt };
