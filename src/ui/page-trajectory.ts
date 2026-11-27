// 页面二：相机轨迹。运行特征提取/匹配/位姿跟踪，展示轨迹、匹配统计与真值对比
import { ReconstructionPipeline } from '../recon/pipeline';
import { evaluateTrajectory } from '../core/evaluate';
import { store } from './store';
import { TopViewMap } from './mapview';
import type { CameraPose } from '../core/types';

let pipeline: ReconstructionPipeline | null = null;
let mapView: TopViewMap | null = null;

export function renderTrajectoryPage(): HTMLElement {
  const el = document.createElement('div');
  el.innerHTML = `
    <div class="card">
      <h2>视觉定位与特征匹配</h2>
      <div class="row">
        <button class="btn" id="runPipelineBtn">运行重建管线</button>
        <span class="pill cpu" id="backendPill">后端检测中…</span>
        <span class="muted small" id="frameInfo"></span>
      </div>
      <div class="progress" style="margin-top:12px"><div id="pipeProgress" style="width:0%"></div></div>
      <div class="muted small" id="pipeDetail" style="margin-top:6px">等待运行</div>
    </div>

    <div class="grid2">
      <div class="card">
        <h2>相机轨迹（俯视）</h2>
        <canvas class="view" id="trajCanvas" width="640" height="520"></canvas>
        <div class="legend" style="margin-top:8px">
          <span><i style="background:#2c4d70"></i>估计轨迹</span>
          <span><i style="background:rgba(110,120,105,.7);border:1px solid #6e7869"></i>真值机位</span>
        </div>
      </div>
      <div class="card">
        <h2>管线统计</h2>
        <div id="stats" class="col"></div>
        <h3>帧间特征与运动掩码</h3>
        <canvas class="view" id="motionCanvas" width="640" height="180"></canvas>
      </div>
    </div>`;

  el.querySelector<HTMLButtonElement>('#runPipelineBtn')!.onclick = () => runPipeline(el);
  refreshStatic(el);
  store.subscribe(() => {
    if (document.body.contains(el)) refreshStatic(el);
  });
  return el;
}

function refreshStatic(el: HTMLElement): void {
  const s = store.get();
  el.querySelector<HTMLElement>('#frameInfo')!.textContent =
    s.renderedFrames.length ? `当前数据：${s.renderedFrames.length} 帧，来源 ${s.source === 'synthetic' ? '合成' : '视频'}` : '请先在导入页准备视频';
  const pill = el.querySelector<HTMLElement>('#backendPill')!;
  pill.className = `pill ${s.computeBackend === 'webgpu' ? 'gpu' : 'cpu'}`;
  pill.textContent = s.computeBackend === 'webgpu' ? 'WebGPU 加速' : 'CPU 回退';
  if (s.result) drawTrajectory(el);
  if (s.observations.length) drawMotionChart(el);
  renderStats(el);
}

async function runPipeline(el: HTMLElement): Promise<void> {
  const s = store.get();
  if (!s.renderedFrames.length) {
    store.toast('请先在导入页渲染或导入视频');
    return;
  }
  const btn = el.querySelector<HTMLButtonElement>('#runPipelineBtn')!;
  btn.disabled = true;
  store.set({ busy: true });
  pipeline = new ReconstructionPipeline();
  const hasGpu = await pipeline.initGpu();

  try {
    const { result, observations } = await pipeline.run(
      {
        frames: s.renderedFrames.map((f) => f.image),
        timesSec: s.renderedFrames.map((f) => f.timeSec),
        fovYDeg: s.fovYDeg,
        frameStep: s.frameStep,
        groundTruth: s.groundTruth ?? undefined,
        poleTruthDiameterM: s.groundTruth?.pole.diameterM ?? 0.03,
      },
      (p) => {
        el.querySelector<HTMLDivElement>('#pipeProgress')!.style.width = `${Math.round(p.ratio * 100)}%`;
        el.querySelector<HTMLElement>('#pipeDetail')!.textContent =
          `${p.stage} · ${p.detail ?? ''}`;
      },
    );

    const accuracy = s.groundTruth
      ? (await import('../core/evaluate')).evaluate(s.groundTruth.culms, result.culms)
      : null;

    store.update((st) => {
      st.result = result;
      st.observations = observations;
      st.computeBackend = hasGpu ? 'webgpu' : 'cpu';
      st.accuracy = accuracy;
      st.busy = false;
      st.matches = [];
    });
    store.toast(`重建完成：${result.culms.length} 根竹竿，${result.poses.length} 个机位`);
    refreshStatic(el);
  } catch (e) {
    store.set({ busy: false });
    btn.disabled = false;
    store.toast(e instanceof Error ? e.message : String(e));
  } finally {
    btn.disabled = false;
  }
}

function drawTrajectory(el: HTMLElement): void {
  const canvas = el.querySelector<HTMLCanvasElement>('#trajCanvas')!;
  const s = store.get();
  if (!mapView) mapView = new TopViewMap(canvas);
  const poses: CameraPose[] = s.result?.poses ?? [];
  mapView.render(
    s.groundTruth?.culms ?? null,
    s.result?.culms ?? [],
    poses,
    {
      showTruth: true,
      showRecon: true,
      showTrajectory: true,
      showLabels: false,
      highlightSpec: null,
      selections: [],
      selectedCulmId: null,
    },
  );
}

function renderStats(el: HTMLElement): void {
  const s = store.get();
  const box = el.querySelector<HTMLElement>('#stats')!;
  const obs = s.observations;
  const result = s.result;
  const totalFeatures = obs.reduce((a, f) => a + f.features.length, 0);
  const totalEdges = obs.reduce((a, f) => a + f.edges.length, 0);
  const avgMotion = obs.length
    ? obs.reduce((a, f) => a + f.motionMaskRatio, 0) / obs.length
    : 0;
  let trajRmse = NaN;
  if (s.groundTruth && result) {
    trajRmse = evaluateTrajectory(
      s.groundTruth.frames.filter((_, i) => i % s.frameStep === 0).slice(0, result.poses.length),
      result.poses,
    );
  }
  const t = result?.timings;
  box.innerHTML = `
    <div class="grid3">
      <div class="metric"><div class="num">${obs.length}</div><div class="label">处理帧</div></div>
      <div class="metric"><div class="num">${totalEdges}</div><div class="label">竖棱观测</div></div>
      <div class="metric"><div class="num">${totalFeatures}</div><div class="label">竹节/斑痕特征</div></div>
      <div class="metric"><div class="num">${(avgMotion * 100).toFixed(1)}%</div><div class="label">风动掩码（剔除）</div></div>
      <div class="metric"><div class="num">${result?.poses.length ?? 0}</div><div class="label">定位机位</div></div>
      <div class="metric ${Number.isFinite(trajRmse) && trajRmse < 0.2 ? 'ok' : ''}">
        <div class="num">${Number.isFinite(trajRmse) ? trajRmse.toFixed(3) : '—'}</div>
        <div class="label">轨迹 RMSE（米）</div>
      </div>
    </div>
    ${t ? `<h3>各阶段耗时（毫秒）</h3>
      <table><tbody>
        <tr><td>特征提取（${s.computeBackend}）</td><td class="mono">${t.gpuFeatureMs.toFixed(0)}</td></tr>
        <tr><td>帧间匹配</td><td class="mono">${t.matchingMs.toFixed(0)}</td></tr>
        <tr><td>相机位姿</td><td class="mono">${t.poseMs.toFixed(0)}</td></tr>
        <tr><td>圆柱拟合（Worker）</td><td class="mono">${t.cylinderFitMs.toFixed(0)}</td></tr>
        <tr><td>竹节聚类</td><td class="mono">${t.nodeMs.toFixed(0)}</td></tr>
        <tr><td>竹龄估计</td><td class="mono">${t.ageMs.toFixed(0)}</td></tr>
        <tr><td><b>管线总耗时</b></td><td class="mono"><b>${(t.totalMs / 1000).toFixed(2)} s</b></td></tr>
      </tbody></table>` : ''}
    ${result ? `<h3>米制定标</h3>
      <p class="small muted">标尺竿估计直径 ${(result.poleDiameterEstimateM * 1000).toFixed(2)} mm，
      真值 ${(result.poleDiameterTruthM * 1000).toFixed(1)} mm，
      尺度校正因子 ${result.scaleCorrection.toFixed(4)}</p>` : ''}`;
}

function drawMotionChart(el: HTMLElement): void {
  const canvas = el.querySelector<HTMLCanvasElement>('#motionCanvas')!;
  const ctx = canvas.getContext('2d')!;
  const obs = store.get().observations;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!obs.length) return;
  const bw = W / obs.length;
  // 运动掩码
  obs.forEach((o, i) => {
    const h = o.motionMaskRatio * (H - 30);
    ctx.fillStyle = '#d98a7a';
    ctx.fillRect(i * bw, H - 20 - h, Math.max(1, bw - 1), h);
  });
  // 特征数折线
  const maxF = Math.max(1, ...obs.map((o) => o.features.length));
  ctx.strokeStyle = '#3f7d3a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  obs.forEach((o, i) => {
    const x = i * bw + bw / 2;
    const y = H - 20 - (o.features.length / maxF) * (H - 30);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#66715f';
  ctx.font = '11px sans-serif';
  ctx.fillText('红：风动掩码占比　绿：每帧特征数', 8, 16);
}
