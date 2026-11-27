// 页面三：竹林重建。精度指标（对照硬性要求）、竹竿列表、单竿侧视、真值对比
import { store } from './store';
import { bendAxisAt } from '../core/grove';
import { ageColor } from './mapview';
import { nodeSpacingsMm } from '../core/selection';
import type { GroundTruthCulm, ReconstructedCulm } from '../core/types';

export function renderReconstructionPage(): HTMLElement {
  const el = document.createElement('div');
  el.innerHTML = `
    <div class="card">
      <h2>重建精度评测</h2>
      <p class="muted small">对照硬性指标：位置 ≤10 cm、胸径 ≤1 mm、节间距 ≤1 cm，且无拆棒/并棒。</p>
      <div class="grid3" id="accuracyMetrics"></div>
    </div>

    <div class="grid2">
      <div class="card">
        <h2>重建竹竿列表</h2>
        <div class="row" style="margin-bottom:8px">
          <input class="field" id="culmFilter" placeholder="按编号筛选…" style="padding:7px 10px;border:1px solid var(--line);border-radius:7px" />
          <span class="muted small" id="culmCount"></span>
        </div>
        <div class="scroll" id="culmTableWrap"></div>
      </div>
      <div class="card">
        <h2>单竿侧视（弯曲与竹节）</h2>
        <canvas class="view" id="sideCanvas" width="560" height="560"></canvas>
        <div id="culmDetail" class="small muted" style="margin-top:8px">点击左侧列表选择竹竿</div>
      </div>
    </div>`;

  el.querySelector<HTMLInputElement>('#culmFilter')!.oninput = () => renderTable(el);
  store.subscribe(() => {
    if (document.body.contains(el)) renderAll(el);
  });
  renderAll(el);
  return el;
}

function renderAll(el: HTMLElement): void {
  renderMetrics(el);
  renderTable(el);
}

function renderMetrics(el: HTMLElement): void {
  const box = el.querySelector<HTMLElement>('#accuracyMetrics')!;
  const acc = store.get().accuracy;
  const result = store.get().result;
  if (!acc) {
    box.innerHTML = `<p class="muted">${result
      ? '当前为真实视频，无真值可评测。合成数据会自动给出完整指标。'
      : '尚未运行重建管线。'}</p>`;
    return;
  }
  const metric = (label: string, value: string, ok: boolean, sub = '') => `
    <div class="metric ${ok ? 'ok' : 'bad'}">
      <div class="num">${value}</div>
      <div class="label">${label}${sub ? `　<span class="small">${sub}</span>` : ''}</div>
    </div>`;
  box.innerHTML = [
    metric('位置 RMSE', `${(acc.positionRmseM * 100).toFixed(2)} cm`, acc.positionRmseM <= 0.1, '要求 ≤10 cm'),
    metric('胸径 RMSE', `${acc.diameterRmseMm.toFixed(2)} mm`, acc.diameterRmseMm <= 1, '要求 ≤1 mm'),
    metric('节间距 RMSE', `${acc.nodeSpacingRmseMm.toFixed(2)} mm`, acc.nodeSpacingRmseMm <= 10, '要求 ≤10 mm'),
    metric('拆棒错误', String(acc.splitErrors), acc.splitErrors === 0, '一竿拆两竿'),
    metric('并棒错误', String(acc.mergeErrors), acc.mergeErrors === 0, '两竿并一竿'),
    metric('竹龄准确率', `${(acc.ageAccuracy * 100).toFixed(0)}%`, acc.ageAccuracy >= 0.8, '±1 年'),
    metric('匹配竿数', `${acc.matchedCulms}/${acc.truthCount}`, acc.matchedCulms >= acc.truthCount * 0.8,
      `重建 ${acc.reconCount}`),
  ].join('');
}

function renderTable(el: HTMLElement): void {
  const wrap = el.querySelector<HTMLElement>('#culmTableWrap')!;
  const culms = store.get().result?.culms ?? [];
  const filter = el.querySelector<HTMLInputElement>('#culmFilter')?.value.trim() ?? '';
  const shown = filter ? culms.filter((c) => String(c.id).includes(filter)) : culms;
  el.querySelector<HTMLElement>('#culmCount')!.textContent = `共 ${culms.length} 根，显示 ${shown.length} 根`;
  if (!culms.length) {
    wrap.innerHTML = '<p class="muted">暂无重建结果</p>';
    return;
  }
  wrap.innerHTML = `
    <table>
      <thead><tr>
        <th>编号</th><th>胸径 mm</th><th>竹龄</th><th>竹节数</th>
        <th>最长节间 mm</th><th>观测帧</th><th>残差 px</th>
      </tr></thead>
      <tbody>
        ${shown.map((c) => `
          <tr data-id="${c.id}" style="cursor:pointer">
            <td><span class="pill info">#${c.id}</span></td>
            <td class="mono">${(c.diameterM * 1000).toFixed(1)}</td>
            <td><span style="color:${ageColor(c.ageYears)};font-weight:600">${c.ageYears.toFixed(1)} 年</span></td>
            <td>${c.nodeHeightsM.length}</td>
            <td class="mono">${nodeSpacingsMm(c).length ? Math.max(...nodeSpacingsMm(c)).toFixed(0) : '—'}</td>
            <td>${c.observationCount}</td>
            <td class="mono">${c.reprojectionErrorPx.toFixed(2)}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
  wrap.querySelectorAll('tr[data-id]').forEach((tr) => {
    tr.addEventListener('click', () => {
      const id = Number(tr.getAttribute('data-id'));
      drawSideView(el, id);
    });
  });
  if (shown.length) drawSideView(el, shown[0].id);
}

function drawSideView(el: HTMLElement, culmId: number): void {
  const s = store.get();
  const culm = s.result?.culms.find((c) => c.id === culmId);
  const canvas = el.querySelector<HTMLCanvasElement>('#sideCanvas')!;
  const detail = el.querySelector<HTMLElement>('#culmDetail')!;
  if (!culm) return;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // 找对应真值
  let truth: GroundTruthCulm | null = null;
  if (s.groundTruth) {
    let best = Infinity;
    for (const t of s.groundTruth.culms) {
      const d = Math.hypot(t.bend.base.x - culm.bend.base.x, t.bend.base.y - culm.bend.base.y);
      if (d < best) {
        best = d;
        truth = t;
      }
    }
    if (best > 0.6) truth = null;
  }

  const maxZ = Math.max(
    culm.nodeHeightsM[culm.nodeHeightsM.length - 1] ?? 6,
    truth?.heightM ?? 0,
    6,
  ) * 1.05;
  const cx = W / 2;
  const groundY = H - 40;
  const zToY = (z: number) => groundY - (z / maxZ) * (H - 80);
  const radiusPx = Math.max(6, culm.diameterM * 900);

  // 地面线
  ctx.strokeStyle = '#9aa593';
  ctx.beginPath();
  ctx.moveTo(30, groundY);
  ctx.lineTo(W - 30, groundY);
  ctx.stroke();

  // 重建竹竿轮廓（弯曲）
  const drawCulm = (
    bend: ReconstructedCulm['bend'], nodes: number[], color: string, dashed: boolean, rPx: number,
  ) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = rPx;
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.beginPath();
    for (let z = 0; z <= maxZ; z += 0.1) {
      const p = bendAxisAt(bend, z);
      const x = cx + (p.x - culm.bend.base.x) * 160;
      const y = zToY(z);
      if (z === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // 竹节
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    nodes.forEach((z) => {
      const p = bendAxisAt(bend, z);
      const x = cx + (p.x - culm.bend.base.x) * 160;
      const y = zToY(z);
      ctx.beginPath();
      ctx.moveTo(x - rPx, y);
      ctx.lineTo(x + rPx, y);
      ctx.stroke();
    });
  };

  drawCulm(culm.bend, culm.nodeHeightsM, '#3f7d3a', false, radiusPx);
  if (truth) {
    drawCulm(truth.bend, truth.nodeHeightsM, 'rgba(176,67,47,0.7)', true, Math.max(5, truth.diameterM * 900));
  }

  // 1.3m 胸径线
  ctx.strokeStyle = '#c28a2e';
  ctx.setLineDash([3, 3]);
  const y13 = zToY(1.3);
  ctx.beginPath();
  ctx.moveTo(30, y13);
  ctx.lineTo(W - 30, y13);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#c28a2e';
  ctx.font = '11px sans-serif';
  ctx.fillText('胸径 1.3 m', 34, y13 - 4);

  // 图例
  ctx.fillStyle = '#3f7d3a';
  ctx.fillRect(W - 150, 14, 14, 4);
  ctx.fillStyle = '#333';
  ctx.fillText('重建', W - 130, 20);
  if (truth) {
    ctx.strokeStyle = 'rgba(176,67,47,0.7)';
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(W - 150, 36);
    ctx.lineTo(W - 136, 36);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#333';
    ctx.fillText('真值', W - 130, 40);
  }

  const spacings = nodeSpacingsMm(culm);
  detail.innerHTML = `
    <b>竹竿 #${culm.id}</b>　胸径 <span class="mono">${(culm.diameterM * 1000).toFixed(2)} mm</span>
    （σ=${(culm.diameterSigmaM * 1000).toFixed(2)}）　竹龄 ${culm.ageYears.toFixed(1)} 年（置信度 ${(culm.ageConfidence * 100).toFixed(0)}%）<br/>
    竹节 ${culm.nodeHeightsM.length} 个，节间距 ${spacings.length ? `${Math.min(...spacings).toFixed(0)}~${Math.max(...spacings).toFixed(0)} mm` : '—'}
    ${truth ? `　真值胸径 ${(truth.diameterM * 1000).toFixed(1)} mm、竹龄 ${truth.ageYears} 年` : ''}`;
}
