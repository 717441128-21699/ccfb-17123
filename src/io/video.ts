// 视频输入：合成场景用 canvas.captureStream + MediaRecorder 导出 WebM；
// 真实视频用 <video> + requestVideoFrameCallback 抽帧（不可用时退化为 setInterval）
export interface CapturedFrame {
  image: ImageData;
  timeSec: number;
}

export class VideoCapture {
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private stream: MediaStream | null = null;

  /** 录制 canvas，返回停止函数（停止后 resolve 出 webm Blob） */
  startCanvasRecording(canvas: HTMLCanvasElement, fps: number, bitsPerSecond = 12_000_000): {
    stop: () => Promise<Blob>;
    mimeType: string;
  } {
    this.chunks = [];
    const fps30 = Math.min(30, fps);
    this.stream = canvas.captureStream(fps30);
    const candidates = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4',
    ];
    const mimeType = candidates.find((c) => MediaRecorder.isTypeSupported(c)) ?? '';
    this.recorder = new MediaRecorder(this.stream, {
      mimeType,
      videoBitsPerSecond: bitsPerSecond,
    });
    this.recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
    this.recorder.start(100);
    const stop = () =>
      new Promise<Blob>((resolve) => {
        if (!this.recorder) {
          resolve(new Blob([], { type: mimeType }));
          return;
        }
        this.recorder.onstop = () => resolve(new Blob(this.chunks, { type: mimeType }));
        this.recorder.stop();
        this.stream?.getTracks().forEach((t) => t.stop());
      });
    return { stop, mimeType };
  }
}

/** 从视频文件按目标帧数等间隔抽帧（缩放到 maxW 以内） */
export async function extractFrames(
  file: File,
  targetFrames: number,
  maxW: number,
  onProgress?: (ratio: number) => void,
): Promise<{ frames: CapturedFrame[]; width: number; height: number; durationSec: number }> {
  const url = URL.createObjectURL(file);
  try {
    const video = document.createElement('video');
    video.src = url;
    video.muted = true;
    video.playsInline = true;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('视频解码失败，请更换文件（建议 H.264/VP9 的 mp4 或 webm）'));
    });
    const duration = video.duration;
    const scale = Math.min(1, maxW / video.videoWidth);
    const w = Math.round(video.videoWidth * scale);
    const h = Math.round(video.videoHeight * scale);
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    const frames: CapturedFrame[] = [];
    const times = Array.from({ length: targetFrames }, (_, i) =>
      Math.min(duration - 0.02, (i + 0.5) * duration / targetFrames));

    for (let i = 0; i < times.length; i++) {
      await seekTo(video, times[i]);
      ctx.drawImage(video, 0, 0, w, h);
      frames.push({ image: ctx.getImageData(0, 0, w, h), timeSec: times[i] });
      onProgress?.((i + 1) / times.length);
    }
    return { frames, width: w, height: h, durationSec: duration };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function seekTo(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve) => {
    const done = () => {
      video.removeEventListener('seeked', done);
      resolve();
    };
    video.addEventListener('seeked', done);
    video.currentTime = t;
  });
}

/** 直接从渲染 canvas 抓一帧 ImageData */
export function frameFromCanvas(canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/** 触发浏览器下载 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export function downloadJson(data: unknown, filename: string): void {
  downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), filename);
}
