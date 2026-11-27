// 页面一：视频导入。支持合成竹林环拍渲染（3 密度 × 3 风力）与真实手机视频导入
import { defaultGenParams, generateGrove } from '../core/grove';
import type { GroveDensity, GroveGenParams, WindLevel } from '../core/types';
import { GroveRenderer } from '../render/renderer';
import { VideoCapture, extractFrames } from '../io/video';
import { store } from './store';

const DENSITY_LABEL: Record<GroveDensity, string> = {
  sparse: '稀疏（48 竿）',
  medium: '中等（96 竿）',
  dense: '茂密（160 竿）',
};
const WIND_LABEL: Record<WindLevel, string> = {
  calm: '无风',
  breeze: '微风',
  windy: '大风（竹叶明显晃动）',
};

export function renderImportPage(): HTMLElement {
  const el = document.createElement('div');
  el.innerHTML = `
    <div class="card">
      <h2>① 合成竹林环拍视频（带真值，可直接评测）</h2>
      <p class="muted small">
        按竹林密度与风力渲染 9 种组合的环拍视频，同时输出每根竹竿的位置、胸径、弯曲、竹节、竹龄真值，
        以及每帧相机位姿。红漆标尺竿直径 30 mm，用于米制定标。
      </p>
      <div class="grid3">
        <label class="field">竹林密度
          <select id="density">
            ${Object.entries(DENSITY_LABEL).map(([k, v]) => `<option value="${k}" ${k === 'medium' ? 'selected' : ''}>${v}</option>`).join('')}
          </select>
        </label>
        <label class="field">风力
          <select id="wind">
            ${Object.entries(WIND_LABEL).map(([k, v]) => `<option value="${k}" ${k === 'breeze' ? 'selected' : ''}>${v}</option>`).join('')}
          </select>
        </label>
        <label class="field">随机种子
          <input id="seed" type="number" value="123" min="1" />
        </label>
        <label class="field">抽帧数量（重建用）
          <input id="frameCount" type="number" value="60" min="24" max="120" />
        </label>
        <label class="field">工作分辨率
          <select id="resolution">
            <option value="640">640×360（快）</option>
            <option value="960" selected>960×540（推荐）</option>
            <option value="1280">1280×720（精细）</option>
          </select>
        </label>
      </div>
      <div class="row" style="margin-top:14px">
        <button class="btn" id="renderBtn">渲染环拍视频</button>
        <button class="btn ghost" id="recordBtn" disabled>导出 WebM 视频</button>
        <button class="btn ghost" id="downloadGt" disabled>下载真值 JSON</button>
        <span class="badge" id="renderStatus">尚未渲染</span>
      </div>
      <div class="progress" style="margin-top:14px"><div id="renderProgress" style="width:0%"></div></div>
      <canvas class="view" id="previewCanvas" style="margin-top:12px;aspect-ratio:16/9"></canvas>
    </div>

    <div class="card">
      <h2>② 导入真实手机视频</h2>
      <p class="muted small">支持 mp4 / webm / mov。导入后按设定帧数等间隔抽帧，建议沿竹林缓步行走环拍一圈。</p>
      <div class="dropzone" id="dropzone">
        点击选择或拖拽视频到此处<br/>
        <span class="small">建议 4K、5 分钟以内，镜头避免剧烈抖动</span>
        <input type="file" id="fileInput" accept="video/*" style="display:none" />
      </div>
      <div class="row" style="margin-top:12px">
        <label class="field">抽帧数量
          <input id="importFrameCount" type="number" value="60" min="24" max="120" />
        </label>
        <span class="badge" id="importStatus">未导入</span>
      </div>
    </div>`;

  el.querySelector<HTMLButtonElement>('#renderBtn')!.onclick = () => renderSynthetic(el);
  el.querySelector<HTMLButtonElement>('#recordBtn')!.onclick = () => recordVideo(el);
  el.querySelector<HTMLButtonElement>('#downloadGt')!.onclick = () => downloadGroundTruth();
  setupDropzone(el);
  return el;
}

async function renderSynthetic(root: HTMLElement): Promise<void> {
  const density = root.querySelector<HTMLSelectElement>('#density')!.value as GroveDensity;
  const wind = root.querySelector<HTMLSelectElement>('#wind')!.value as WindLevel;
  const seed = Number(root.querySelector<HTMLInputElement>('#seed')!.value);
  const targetFrames = Number(root.querySelector<HTMLInputElement>('#frameCount')!.value);
  const workW = Number(root.querySelector<HTMLSelectElement>('#resolution')!.value);
  const workH = Math.round(workW * 9 / 16);

  const base = defaultGenParams(density, wind, seed);
  const params: GroveGenParams = {
    ...base,
    frameCount: Math.max(120, targetFrames * 4), // 渲染密、重建抽样稀
    width: workW,
    height: workH,
  };
  const gt = generateGrove(params);
  const canvas = root.querySelector<HTMLCanvasElement>('#previewCanvas')!;
  canvas.width = workW;
  canvas.height = workH;

  store.set({
    busy: true,
    groundTruth: gt,
    source: 'synthetic',
    fovYDeg: params.fovYDeg,
    renderedFrames: [],
    result: null,
    accuracy: null,
    selections: [],
  });

  const renderer = new GroveRenderer(canvas, gt);
  const step = Math.max(1, Math.round(params.frameCount / targetFrames));
  const frames: { image: ImageData; timeSec: number }[] = [];
  const progress = root.querySelector<HTMLDivElement>('#renderProgress')!;
  const status = root.querySelector<HTMLElement>('#renderStatus')!;
  const fps = 30;

  try {
    for (let i = 0; i < params.frameCount; i++) {
      const timeSec = i / fps;
      renderer.renderFrame(i, timeSec);
      if (i % step === 0) {
        const px = renderer.readPixels();
        frames.push({
          image: new ImageData(new Uint8ClampedArray(px), workW, workH),
          timeSec,
        });
      }
      if (i % 6 === 0) {
        progress.style.width = `${((i + 1) / params.frameCount) * 100}%`;
        status.textContent = `渲染中 ${i + 1}/${params.frameCount}`;
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    store.update((s) => {
      s.renderedFrames = frames;
      s.frameStep = step;
      s.busy = false;
    });
    progress.style.width = '100%';
    status.textContent = `完成：${gt.culms.length} 竿，抽样 ${frames.length} 帧（风力 ${wind}）`;
    root.querySelector<HTMLButtonElement>('#recordBtn')!.disabled = false;
    root.querySelector<HTMLButtonElement>('#downloadGt')!.disabled = false;
    store.toast(`合成视频渲染完成，${frames.length} 帧已就绪，可进入「相机轨迹」页`);
  } catch (e) {
    store.set({ busy: false });
    status.textContent = '渲染失败';
    store.toast(e instanceof Error ? e.message : String(e));
  }
}

async function recordVideo(root: HTMLElement): Promise<void> {
  const gt = store.get().groundTruth;
  const canvas = root.querySelector<HTMLCanvasElement>('#previewCanvas')!;
  if (!gt) return;
  const renderer = new GroveRenderer(canvas, gt);
  const capture = new VideoCapture();
  const { stop, mimeType } = capture.startCanvasRecording(canvas, 30);
  for (let i = 0; i < gt.params.frameCount; i++) {
    renderer.renderFrame(i, i / 30);
    await new Promise((r) => setTimeout(r, 1000 / 30));
  }
  const blob = await stop();
  store.update((s) => {
    s.videoBlob = blob;
    s.videoFileName = `grove-${gt.params.density}-${gt.params.wind}.${mimeType.includes('mp4') ? 'mp4' : 'webm'}`;
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = store.get().videoFileName;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  store.toast('WebM 视频已导出并开始下载');
}

function downloadGroundTruth(): void {
  const gt = store.get().groundTruth;
  if (!gt) return;
  const blob = new Blob([JSON.stringify(gt, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `grove-truth-${gt.params.density}-${gt.params.wind}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function setupDropzone(root: HTMLElement): void {
  const dz = root.querySelector<HTMLDivElement>('#dropzone')!;
  const input = root.querySelector<HTMLInputElement>('#fileInput')!;
  const status = root.querySelector<HTMLElement>('#importStatus')!;
  dz.onclick = () => input.click();
  dz.ondragover = (e) => {
    e.preventDefault();
    dz.classList.add('drag');
  };
  dz.ondragleave = () => dz.classList.remove('drag');
  dz.ondrop = (e) => {
    e.preventDefault();
    dz.classList.remove('drag');
    const file = e.dataTransfer?.files?.[0];
    if (file) handleVideoFile(file, status, root);
  };
  input.onchange = () => {
    const file = input.files?.[0];
    if (file) handleVideoFile(file, status, root);
  };
}

async function handleVideoFile(
  file: File, status: HTMLElement, root: HTMLElement,
): Promise<void> {
  const targetFrames = Number(root.querySelector<HTMLInputElement>('#importFrameCount')!.value);
  status.textContent = '解码抽帧中…';
  store.set({ busy: true });
  try {
    const { frames, width, height, durationSec } = await extractFrames(
      file, targetFrames, 1280,
      (ratio) => {
        status.textContent = `抽帧 ${Math.round(ratio * 100)}%`;
      },
    );
    const step = Math.max(1, Math.round(durationSec * 30 / targetFrames));
    store.update((s) => {
      s.source = 'video';
      s.renderedFrames = frames;
      s.videoFileName = file.name;
      s.frameStep = step;
      s.groundTruth = null;
      s.result = null;
      s.accuracy = null;
      s.selections = [];
      s.busy = false;
      s.fovYDeg = 55;
    });
    status.textContent = `${file.name}：${width}×${height}，${durationSec.toFixed(1)}s，${frames.length} 帧`;
    store.toast(`视频导入成功，${frames.length} 帧已就绪`);
  } catch (e) {
    store.set({ busy: false });
    status.textContent = '导入失败';
    store.toast(e instanceof Error ? e.message : String(e));
  }
}
