// 全局应用状态：跨页面共享的数据集、观测、重建结果与选竹结果
import type {
  AccuracyReport, BambooSelection, CameraPose, FeatureMatch,
  FrameObservations, GroveGroundTruth, PipelineTimings, ReconstructionResult,
} from '../core/types';

export interface RenderedFrameInfo {
  image: ImageData;
  timeSec: number;
}

export interface AppState {
  // 数据集
  groundTruth: GroveGroundTruth | null;
  renderedFrames: RenderedFrameInfo[];
  videoBlob: Blob | null;
  videoFileName: string;
  source: 'synthetic' | 'video' | null;
  fovYDeg: number;
  frameStep: number;
  // 视觉管线
  observations: FrameObservations[];
  matches: FeatureMatch[];
  computeBackend: 'webgpu' | 'cpu';
  // 重建
  result: ReconstructionResult | null;
  accuracy: AccuracyReport | null;
  // 选竹
  selections: BambooSelection[];
  selectedSpecKey: string;
  // 运行状态
  busy: boolean;
  progressStage: string;
  progressRatio: number;
  progressDetail: string;
  toast: string | null;
}

export const initialState: AppState = {
  groundTruth: null,
  renderedFrames: [],
  videoBlob: null,
  videoFileName: '',
  source: null,
  fovYDeg: 55,
  frameStep: 5,
  observations: [],
  matches: [],
  computeBackend: 'cpu',
  result: null,
  accuracy: null,
  selections: [],
  selectedSpecKey: 'G',
  busy: false,
  progressRatio: 0,
  progressStage: '',
  progressDetail: '',
  toast: null,
};

type Listener = (state: AppState) => void;

/** 极简发布订阅 store（无框架依赖） */
export class Store {
  private state: AppState;
  private listeners = new Set<Listener>();

  constructor(init: AppState) {
    this.state = structuredClone(init);
  }

  get(): AppState {
    return this.state;
  }

  set(patch: Partial<AppState>): void {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((l) => l(this.state));
  }

  update(fn: (draft: AppState) => void): void {
    fn(this.state);
    this.state = { ...this.state };
    this.listeners.forEach((l) => l(this.state));
  }

  subscribe(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  toast(message: string, durationMs = 3200): void {
    this.set({ toast: message });
    setTimeout(() => {
      if (this.get().toast === message) this.set({ toast: null });
    }, durationMs);
  }

  reset(): void {
    this.state = structuredClone(initialState);
    this.listeners.forEach((l) => l(this.state));
  }
}

export const store = new Store(initialState);

export function posesOf(result: ReconstructionResult | null): CameraPose[] {
  return result?.poses ?? [];
}

export function timingsOf(result: ReconstructionResult | null): PipelineTimings | null {
  return result?.timings ?? null;
}
