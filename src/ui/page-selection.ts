// 页面四：选竹标注。按调门规格在俯视图标号，支持编辑规格、查看理由、导出清单
import { DEFAULT_FLUTE_SPECS } from '../core/specs';
import { selectCulms, nodeSpacingsMm } from '../core/selection';
import { store } from './store';
import { TopViewMap, specColor } from './mapview';
import type { FluteSpec } from '../core/types';

let mapView: TopViewMap | null = null;
let specs: FluteSpec[] = DEFAULT_FLUTE_SPECS.map((s) => ({ ...s }));
let selectedCulm: number | null = null;

export function renderSelectionPage(): HTMLElement {
  const el = document.createElement('div');
  el.innerHTML = `
    <div class="card">
      <h2>调门用料规格（可编辑）</h2>
      <div class="scroll">
        <table id="specTable">
          <thead><tr>
            <th>调门</th><th>目标胸径 mm</th><th>容差 mm</th>
            <th>最短节间 mm</th><th>最低竹龄</th><th></th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="row" style="margin-top:12px">
        <button class="btn" id="applySpecBtn">重新选竹</button>
        <button class="btn ghost" id="resetSpecBtn">恢复默认规格</button>
        <button class="btn ghost" id="exportBtn">导出选竹清单 JSON</button>
      </div>
    </div>

    <div class="grid2">
      <div class="card">
        <h2>俯视图标号</h2>
        <div class="row" id="specFilter" style="margin-bottom:10px"></div>
        <canvas class="view" id="selectMap" width="640" height="560"></canvas>
        <div class="legend" style="margin-top:8px" id="mapLegend"></div>
      </div>
      <div class="card">
        <h2>选中竹竿与适配理由</h2>
        <div id="selectionList" class="scroll"></div>
      </div>
    </div>`;

  el.querySelector<HTMLButtonElement>('#applySpecBtn')!.onclick = () => applySelection(el);
  el.querySelector<HTMLButtonElement>('#resetSpecBtn')!.onclick = () => {
    specs = DEFAULT_FLUTE_SPECS.map((s) => ({ ...s }));
    renderSpecTable(el);
    applySelection(el);
  };
  el.querySelector<HTMLButtonElement>('#exportBtn')!.onclick = () => exportSelections();

  renderSpecTable(el);
  buildSpecFilter(el);
  store.subscribe(() => {
    if (document.body.contains(el)) refreshMap(el);
  });
  applySelection(el);
  return el;
}

function renderSpecTable(el: HTMLElement): void {
  const tbody = el.querySelector<HTMLTableSectionElement>('#specTable tbody')!;
  tbody.innerHTML = specs.map((s, i) => `
    <tr>
      <td><b>${s.key}</b> <span class="muted small">${s.label.split('（')[1]?.replace('）', '') ?? ''}</span></td>
      <td><input data-i="${i}" data-k="diameterMm" value="${s.diameterMm}" type="number" step="0.1" style="width:80px"></td>
      <td><input data-i="${i}" data-k="diameterTolMm" value="${s.diameterTolMm}" type="number" step="0.1" style="width:70px"></td>
      <td><input data-i="${i}" data-k="minNodeSpacingMm" value="${s.minNodeSpacingMm}" type="number" style="width:80px"></td>
      <td><input data-i="${i}" data-k="minAge" value="${s.minAge}" type="number" min="1" max="6" style="width:60px"></td>
      <td class="muted small">${s.label}</td>
    </tr>`).join('');
  tbody.querySelectorAll<HTMLInputElement>('input').forEach((input) => {
    input.onchange = () => {
      const i = Number(input.dataset.i);
      const k = input.dataset.k as keyof FluteSpec;
      (specs[i] as unknown as Record<string, number>)[k] = Number(input.value);
    };
  });
}

function buildSpecFilter(el: HTMLElement): void {
  const box = el.querySelector<HTMLElement>('#specFilter')!;
  box.innerHTML = `<button class="seg active" data-key="">全部调门</button>` +
    specs.map((s) => `<button class="seg" data-key="${s.key}">${s.key} 调</button>`).join('');
  box.querySelectorAll('button').forEach((btn) => {
    btn.onclick = () => {
      box.querySelectorAll('button').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      store.set({ selectedSpecKey: btn.dataset.key ?? '' });
      refreshMap(el);
      renderSelectionList(el);
    };
  });
}

function applySelection(el: HTMLElement): void {
  const culms = store.get().result?.culms ?? [];
  if (!culms.length) {
    store.set({ selections: [] });
    store.toast('请先在相机轨迹页运行重建管线');
    refreshMap(el);
    renderSelectionList(el);
    return;
  }
  const selections = selectCulms(culms, specs);
  store.set({ selections });
  refreshMap(el);
  renderSelectionList(el);
  store.toast(`选出 ${new Set(selections.map((s) => s.culmId)).size} 根合用竹竿`);
}

function refreshMap(el: HTMLElement): void {
  const canvas = el.querySelector<HTMLCanvasElement>('#selectMap')!;
  const s = store.get();
  if (!mapView) mapView = new TopViewMap(canvas);
  const highlight = s.selectedSpecKey || null;
  mapView.render(
    s.groundTruth?.culms ?? null,
    s.result?.culms ?? [],
    s.result?.poses ?? [],
    {
      showTruth: false,
      showRecon: true,
      showTrajectory: false,
      showLabels: true,
      highlightSpec: highlight,
      selections: s.selections,
      selectedCulmId: selectedCulm,
      onSelect: (id) => {
        selectedCulm = id;
        refreshMap(el);
        renderSelectionList(el, id ?? undefined);
      },
    },
  );
  const legend = el.querySelector<HTMLElement>('#mapLegend')!;
  legend.innerHTML = specs.map((sp) =>
    `<span><i style="background:${specColor(sp.key)}"></i>${sp.key} 调</span>`).join('') +
    `<span><i style="background:rgb(150,145,72)"></i>未入选</span>`;
}

function renderSelectionList(el: HTMLElement, focusId?: number): void {
  const box = el.querySelector<HTMLElement>('#selectionList')!;
  const s = store.get();
  const filter = s.selectedSpecKey;
  let list = s.selections.slice().sort((a, b) => b.score - a.score);
  if (filter) list = list.filter((x) => x.specKey === filter);
  if (focusId !== undefined) {
    const focused = list.find((x) => x.culmId === focusId);
    if (focused) list = [focused, ...list.filter((x) => x !== focused)];
  }
  if (!list.length) {
    box.innerHTML = '<p class="muted">当前调门没有合用竹竿，可放宽规格后重新选竹。</p>';
    return;
  }
  const culmById = new Map((s.result?.culms ?? []).map((c) => [c.id, c]));
  box.innerHTML = list.map((sel) => {
    const c = culmById.get(sel.culmId);
    if (!c) return '';
    const sp = nodeSpacingsMm(c);
    return `
      <div class="metric" style="margin-bottom:8px;border-left:4px solid ${specColor(sel.specKey)}">
        <div class="row" style="justify-content:space-between">
          <b>#${sel.culmId}　${sel.specKey} 调</b>
          <span class="pill gpu">评分 ${(sel.score * 100).toFixed(0)}</span>
        </div>
        <div class="small muted" style="margin-top:4px">
          胸径 ${(c.diameterM * 1000).toFixed(1)} mm ｜ 竹龄 ${c.ageYears.toFixed(1)} 年 ｜
          最长节间 ${sp.length ? Math.max(...sp).toFixed(0) : '—'} mm
        </div>
        <ul class="small" style="margin:6px 0 0;padding-left:18px">
          ${sel.reasons.map((r) => `<li>${r}</li>`).join('')}
        </ul>
      </div>`;
  }).join('');
}

function exportSelections(): void {
  const s = store.get();
  if (!s.selections.length) {
    store.toast('没有可导出的选竹结果');
    return;
  }
  const payload = {
    exportedAt: new Date().toISOString(),
    source: s.source,
    specs,
    selections: s.selections,
    culms: (s.result?.culms ?? [])
      .filter((c) => s.selections.some((x) => x.culmId === c.id))
      .map((c) => ({
        id: c.id,
        position: c.bend.base,
        diameterMm: +(c.diameterM * 1000).toFixed(2),
        nodeHeightsMm: c.nodeHeightsM.map((z) => +(z * 1000).toFixed(1)),
        ageYears: +c.ageYears.toFixed(1),
      })),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'bamboo-selection.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  store.toast('选竹清单已导出');
}
