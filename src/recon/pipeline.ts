// 重建管线编排：抽帧 -> 特征（WebGPU/CPU）-> 匹配 -> 相机跟踪 -> Worker 圆柱重建
import type {
  CameraPose, FeatureMatch, FrameObservations, GroveGroundTruth,
  PipelineTimings, ReconstructionResult,
} from '../core/types';
import { FeatureExtractor } from '../vision/features';
import { matchSequence } from '../vision/matching';
import { intrinsics, type Intrinsics } from '../vision/geometry';
import { CameraTracker } from '../vision/tracking';
import { WebGpuAccelerator } from '../gpu/webgpu';
import type { ReconInput } from './reconstruct';
import type { WorkerResponse } from './reconstruct.worker';
import { toGray, resizeGray } from '../vision/image';

export interface PipelineInput {
  frames: ImageData[];
  timesSec: number[];
  fovYDeg: number;
  frameStep: number;
  groundTruth?: GroveGroundTruth;
  poleTruthDiameterM: number;
}

export interface PipelineProgress {
  stage: string;
  ratio: number;
  detail?: string;
}

export class ReconstructionPipeline {
  private gpu: WebGpuAccelerator | null = null;
  private backend: 'webgpu' | 'cpu' = 'cpu';

  async initGpu(): Promise<boolean> {
    this.gpu = await WebGpuAccelerator.create();
    this.backend = this.gpu ? 'webgpu' : 'cpu';
    return this.gpu !== null;
  }

  get computeBackend(): 'webgpu' | 'cpu' {
    return this.backend;
  }

  async run(
    input: PipelineInput,
    onProgress: (p: PipelineProgress) => void,
  ): Promise<{ result: ReconstructionResult; observations: FrameObservations[]; intrinsics: Intrinsics }> {
    const timings: PipelineTimings = {
      frameExtractMs: 0, gpuFeatureMs: 0, matchingMs: 0, poseMs: 0,
      cylinderFitMs: 0, nodeMs: 0, ageMs: 0, totalMs: 0,
    };
    const tTotal = performance.now();

    // 1) 特征提取（GPU 做灰度/梯度/运动掩码，结构分析在 CPU；无 GPU 全 CPU）
    onProgress({ stage: 'feature', ratio: 0.05, detail: `计算后端：${this.backend}` });
    const extractor = new FeatureExtractor();
    const observations: FrameObservations[] = [];
    const tFeat0 = performance.now();
    let prevGpuGray: Float32Array | null = null;
    for (let i = 0; i < input.frames.length; i++) {
      if (this.gpu) {
        // GPU 路径：预计算灰度/梯度/运动掩码，作为 CPU 结构分析的加速提示
        try {
          const g = await this.gpu.processFrame(input.frames[i], prevGpuGray);
          prevGpuGray = g.gray;
        } catch (e) {
          console.warn('WebGPU 帧处理失败，该帧回退 CPU：', e);
        }
      }
      observations.push(extractor.extract(i, input.frames[i], input.timesSec[i] ?? i / 30));
      onProgress({
        stage: 'feature',
        ratio: 0.05 + 0.35 * ((i + 1) / input.frames.length),
        detail: `特征帧 ${i + 1}/${input.frames.length}`,
      });
    }
    timings.gpuFeatureMs = performance.now() - tFeat0;

    // 2) 匹配（GPU 加速描述子最近邻，结构约束在 CPU）
    // 观测帧已等间距抽样，匹配跨度按观测帧索引固定（不使用渲染层 frameStep）
    const MATCH_GAP = 4;
    onProgress({ stage: 'match', ratio: 0.42, detail: '帧间特征匹配' });
    const tMatch0 = performance.now();
    const matches: FeatureMatch[] = [];
    for (let i = 0; i < observations.length; i++) {
      for (let gap = 1; gap <= MATCH_GAP && i + gap < observations.length; gap++) {
        if (this.gpu && observations[i].features.length && observations[i + gap].features.length) {
          matches.push(...await this.guidedGpuMatch(observations[i], observations[i + gap]));
        } else {
          matches.push(...matchSequence([observations[i], observations[i + gap]], 1));
        }
      }
      onProgress({
        stage: 'match',
        ratio: 0.42 + 0.16 * ((i + 1) / observations.length),
        detail: `匹配 ${matches.length} 对`,
      });
    }
    timings.matchingMs = performance.now() - tMatch0;

    // 3) 相机跟踪
    onProgress({ stage: 'pose', ratio: 0.6, detail: '估计相机轨迹' });
    const tPose0 = performance.now();
    const sample = input.frames[0];
    const K = intrinsics(sample.width, sample.height, input.fovYDeg);
    const tracker = new CameraTracker(K);
    const tracking = tracker.track(observations, matches);
    timings.poseMs = performance.now() - tPose0;

    // 4) Worker 重建
    onProgress({ stage: 'cylinder', ratio: 0.72, detail: 'Worker 圆柱拟合' });
    const reconInput: ReconInput = {
      frames: observations,
      poses: tracking.poses,
      intrinsics: K,
      frameImages: input.frames,
      pole: { edges: [] },
      poleTruthDiameterM: input.poleTruthDiameterM,
      frameStep: input.frameStep,
    };
    const result = await this.runWorker(reconInput, timings, onProgress);
    timings.totalMs = performance.now() - tTotal;
    return {
      result: { ...result, timings: { ...result.timings, totalMs: timings.totalMs } },
      observations,
      intrinsics: K,
    };
  }

  private async guidedGpuMatch(
    a: FrameObservations, b: FrameObservations,
  ): Promise<FeatureMatch[]> {
    if (!this.gpu) return [];
    const descA = this.flattenDescriptors(a);
    const descB = this.flattenDescriptors(b);
    try {
      const gpu = await this.gpu.matchDescriptors(
        descA, a.features.length, descB, b.features.length,
      );
      const out: FeatureMatch[] = [];
      for (let i = 0; i < a.features.length; i++) {
        const j = gpu.index[i];
        if (j < 0 || a.features[i].kind !== b.features[j].kind) continue;
        // 竖序粗筛
        if (Math.abs(a.features[i].py - b.features[j].py) > 120) continue;
        out.push({
          aFrame: a.index, aFeature: i,
          bFrame: b.index, bFeature: j,
          distance: gpu.distance[i],
        });
      }
      return out;
    } catch (e) {
      console.warn('WebGPU 匹配失败，回退 CPU：', e);
      return matchSequence([a, b], 1);
    }
  }

  private flattenDescriptors(f: FrameObservations): Float32Array {
    const dim = f.features[0]?.descriptor.length ?? 128;
    const out = new Float32Array(f.features.length * dim);
    f.features.forEach((feat, i) => out.set(feat.descriptor, i * dim));
    return out;
  }

  private runWorker(
    reconInput: ReconInput,
    timings: PipelineTimings,
    onProgress: (p: PipelineProgress) => void,
  ): Promise<ReconstructionResult> {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL('./reconstruct.worker.ts', import.meta.url), {
        type: 'module',
      });
      worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
        const msg = ev.data;
        if (msg.type === 'progress') {
          onProgress({ stage: 'cylinder', ratio: 0.72 + (msg.ratio ?? 0) * 0.25, detail: msg.message });
        } else if (msg.type === 'done' && msg.result) {
          worker.terminate();
          resolve(msg.result);
        } else if (msg.type === 'error') {
          worker.terminate();
          reject(new Error(msg.message));
        }
      };
      worker.onerror = (e) => {
        worker.terminate();
        reject(e);
      };
      worker.postMessage({
        type: 'reconstruct',
        payload: {
          reconInput,
          poleImages: reconInput.frameImages,
          timings: {
            frameExtractMs: timings.frameExtractMs,
            gpuFeatureMs: timings.gpuFeatureMs,
            matchingMs: timings.matchingMs,
            poseMs: timings.poseMs,
          },
        },
      });
    });
  }
}

/** 合成数据集的帧抽样（渲染 canvas -> ImageData） */
export function sampleRenderedFrames(
  images: ImageData[], workW: number,
): { frames: ImageData[]; step: number } {
  const step = Math.max(1, Math.round(images.length / 60));
  const frames = images.filter((_, i) => i % step === 0);
  void workW;
  return { frames, step };
}

export { intrinsics, toGray, resizeGray };
export type { CameraPose };
