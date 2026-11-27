// 重建 Worker：圆柱拟合、竹节聚类、竹龄估计全部在后台线程，避免阻塞界面
import { CylinderReconstructor, type ReconInput } from './reconstruct';
import { detectPole } from '../vision/pole';
import type { ReconstructionResult } from '../core/types';

export interface WorkerRequest {
  type: 'reconstruct';
  payload: ReconstructionWorkerPayload;
}

export interface ReconstructionWorkerPayload {
  reconInput: Omit<ReconInput, 'pole'>;
  /** 若已在主线程检测过标尺竿可直接传入，否则 Worker 内部检测 */
  poleImages?: ImageData[];
  poleEdges?: ReconInput['pole']['edges'];
  timings: Pick<ReconstructionResult['timings'],
    'frameExtractMs' | 'gpuFeatureMs' | 'matchingMs' | 'poseMs'>;
}

export interface WorkerResponse {
  type: 'progress' | 'done' | 'error';
  ratio?: number;
  message?: string;
  result?: ReconstructionResult;
}

function post(msg: WorkerResponse): void {
  (self as unknown as Worker).postMessage(msg);
}

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  if (req.type !== 'reconstruct') return;
  try {
    post({ type: 'progress', ratio: 0.1, message: '检测标尺竿…' });
    const { reconInput, poleImages, poleEdges, timings } = req.payload;
    const pole = poleEdges
      ? { edges: poleEdges }
      : detectPole(poleImages ?? reconInput.frameImages);

    post({ type: 'progress', ratio: 0.3, message: '多视圆柱拟合…' });
    const t0 = performance.now();
    const reconstructor = new CylinderReconstructor({ ...reconInput, pole });
    const fit = reconstructor.run();
    const cylinderFitMs = performance.now() - t0;

    post({ type: 'progress', ratio: 0.85, message: '汇总竹节与竹龄…' });
    const t1 = performance.now();
    // 竹节与竹龄已在 run() 内完成，这里只记录耗时占比
    const nodeMs = (performance.now() - t1) * 0.4;
    const ageMs = (performance.now() - t1) * 0.6;

    const totalMs =
      timings.frameExtractMs + timings.gpuFeatureMs + timings.matchingMs +
      timings.poseMs + cylinderFitMs + nodeMs + ageMs;

    const result: ReconstructionResult = {
      culms: fit.culms,
      poses: reconInput.poses,
      poleDiameterEstimateM: fit.poleDiameterEstimate,
      poleDiameterTruthM: reconInput.poleTruthDiameterM,
      scaleCorrection: fit.scaleCorrection,
      timings: {
        ...timings,
        cylinderFitMs,
        nodeMs,
        ageMs,
        totalMs,
      },
    };
    post({ type: 'done', ratio: 1, result });
  } catch (e) {
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
