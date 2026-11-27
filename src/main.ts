import './ui/styles.css';
import { store } from './ui/store';
import { renderImportPage } from './ui/page-import';
import { renderTrajectoryPage } from './ui/page-trajectory';
import { renderReconstructionPage } from './ui/page-reconstruction';
import { renderSelectionPage } from './ui/page-selection';

type PageId = 'import' | 'trajectory' | 'reconstruction' | 'selection';

const TABS: { id: PageId; label: string }[] = [
  { id: 'import', label: '视频导入' },
  { id: 'trajectory', label: '相机轨迹' },
  { id: 'reconstruction', label: '竹林重建' },
  { id: 'selection', label: '选竹标注' },
];

const renderers: Record<PageId, () => HTMLElement> = {
  import: renderImportPage,
  trajectory: renderTrajectoryPage,
  reconstruction: renderReconstructionPage,
  selection: renderSelectionPage,
};

let currentPage: PageId = 'import';

function mount(page: PageId): void {
  currentPage = page;
  const main = document.querySelector<HTMLElement>('main')!;
  main.innerHTML = '';
  main.appendChild(renderers[page]());
  document.querySelectorAll<HTMLButtonElement>('nav.tabs button').forEach((b) => {
    b.classList.toggle('active', b.dataset.page === page);
  });
}

function boot(): void {
  const header = document.querySelector<HTMLElement>('header.topbar')!;
  const nav = header.querySelector<HTMLElement>('nav.tabs')!;
  nav.innerHTML = TABS.map((t) =>
    `<button data-page="${t.id}" class="${t.id === currentPage ? 'active' : ''}">${t.label}</button>`).join('');
  nav.querySelectorAll('button').forEach((b) => {
    b.onclick = () => mount(b.dataset.page as PageId);
  });

  // 全局 toast
  const toastEl = document.createElement('div');
  toastEl.className = 'toast';
  toastEl.hidden = true;
  document.body.appendChild(toastEl);
  store.subscribe((s) => {
    toastEl.hidden = !s.toast;
    toastEl.textContent = s.toast ?? '';
  });

  mount(currentPage);

  // 全局 WebGPU 能力检测
  const badge = document.querySelector<HTMLElement>('#globalBadge')!;
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    navigator.gpu.requestAdapter().then((a) => {
      badge.textContent = a ? 'WebGPU 可用' : 'WebGPU 无适配器，将用 CPU';
      badge.style.background = a ? 'rgba(47,125,79,0.55)' : 'rgba(176,67,47,0.55)';
    }).catch(() => {
      badge.textContent = 'WebGPU 不可用，将用 CPU';
      badge.style.background = 'rgba(176,67,47,0.55)';
    });
  } else {
    badge.textContent = 'WebGPU 不可用，CPU 回退';
    badge.style.background = 'rgba(176,67,47,0.55)';
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
