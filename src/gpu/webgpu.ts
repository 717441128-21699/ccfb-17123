// WebGPU compute 加速层：灰度、Sobel、运动掩码、描述子最近邻匹配
// 无 WebGPU 或内核失败时由 pipeline 自动回退到 CPU（vision/ 下的同算法实现）
import { DESCRIPTOR_DIM } from '../vision/features';

const WGSL = /* wgsl */ `
struct Dim { w: u32, h: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> dim: Dim;
@group(0) @binding(1) var<storage, read> rgba: array<u32>;
@group(0) @binding(2) var<storage, read_write> gray: array<f32>;
@group(0) @binding(3) var<storage, read_write> gx: array<f32>;
@group(0) @binding(4) var<storage, read_write> gy: array<f32>;
@group(0) @binding(5) var<storage, read_write> mag: array<f32>;
@group(0) @binding(6) var<storage, read> prevGray: array<f32>;
@group(0) @binding(7) var<storage, read_write> motion: array<u32>;

fn unpack(c: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(c & 255u),
    f32((c >> 8u) & 255u),
    f32((c >> 16u) & 255u),
    f32((c >> 24u) & 255u)
  );
}

@compute @workgroup_size(16, 16)
fn grayKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= dim.w || y >= dim.h) { return; }
  let i = y * dim.w + x;
  let c = unpack(rgba[i]);
  gray[i] = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

@compute @workgroup_size(16, 16)
fn sobelKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x < 1u || y < 1u || x >= dim.w - 1u || y >= dim.h - 1u) { return; }
  let i = y * dim.w + x;
  let tl = gray[i - dim.w - 1u]; let t = gray[i - dim.w]; let tr = gray[i - dim.w + 1u];
  let l = gray[i - 1u]; let r = gray[i + 1u];
  let bl = gray[i + dim.w - 1u]; let b = gray[i + dim.w]; let br = gray[i + dim.w + 1u];
  let sx = tr + 2.0 * r + br - tl - 2.0 * l - bl;
  let sy = bl + 2.0 * b + br - tl - 2.0 * t - tr;
  gx[i] = sx; gy[i] = sy; mag[i] = sqrt(sx * sx + sy * sy);
}

@compute @workgroup_size(16, 16)
fn motionKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= dim.w || y >= dim.h) { return; }
  let i = y * dim.w + x;
  var hit = 0u;
  for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
    for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
      let xx = i32(x) + dx; let yy = i32(y) + dy;
      if (xx >= 0 && yy >= 0 && xx < i32(dim.w) && yy < i32(dim.h)) {
        let j = u32(yy) * dim.w + u32(xx);
        if (abs(gray[j] - prevGray[j]) > 14.0) { hit = 1u; }
      }
    }
  }
  motion[i] = hit;
}

// ---------- 描述子匹配：每个线程处理一个 A 描述子，遍历 B ----------
struct MatchCfg { countA: u32, countB: u32, ratio: f32, maxDist: f32 };
@group(0) @binding(0) var<uniform> mcfg: MatchCfg;
@group(0) @binding(1) var<storage, read> descA: array<f32>;
@group(0) @binding(2) var<storage, read> descB: array<f32>;
@group(0) @binding(3) var<storage, read_write> bestIdx: array<i32>;
@group(0) @binding(4) var<storage, read_write> bestDist: array<f32>;

@compute @workgroup_size(64)
fn matchKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let a = gid.x;
  if (a >= mcfg.countA) { return; }
  var best = 1e9f32; var second = 1e9f32; var bi = -1i;
  for (var b = 0u; b < mcfg.countB; b = b + 1u) {
    var s = 0.0f32;
    for (var k = 0u; k < ${DESCRIPTOR_DIM}u; k = k + 1u) {
      let d = descA[a * ${DESCRIPTOR_DIM}u + k] - descB[b * ${DESCRIPTOR_DIM}u + k];
      s = s + d * d;
    }
    if (s < best) { second = best; best = s; bi = i32(b); }
    else if (s < second) { second = s; }
  }
  let rootBest = sqrt(best);
  if (rootBest < mcfg.maxDist && rootBest < mcfg.ratio * sqrt(second)) {
    bestIdx[a] = bi;
    bestDist[a] = rootBest;
  } else {
    bestIdx[a] = -1i;
  }
}
`;

export interface GpuFrameBuffers {
  gray: Float32Array;
  gx: Float32Array;
  gy: Float32Array;
  mag: Float32Array;
  motion: Uint8Array;
  motionRatio: number;
}

export class WebGpuAccelerator {
  private constructor(
    private device: GPUDevice,
    private grayPipeline: GPUComputePipeline,
    private sobelPipeline: GPUComputePipeline,
    private motionPipeline: GPUComputePipeline,
    private matchPipeline: GPUComputePipeline,
  ) {}

  static async create(): Promise<WebGpuAccelerator | null> {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null;
    let device: GPUDevice;
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return null;
      device = await adapter.requestDevice();
      device.addEventListener('uncapturederror', (e) =>
        console.warn('WebGPU 运行错误（将回退 CPU）：', (e as GPUUncapturedErrorEvent).error));
    } catch {
      return null;
    }
    const module = device.createShaderModule({ code: WGSL });
    const mk = (entry: string) =>
      device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: entry },
      });
    try {
      const [grayPipeline, sobelPipeline, motionPipeline, matchPipeline] = await Promise.all([
        mk('grayKernel'), mk('sobelKernel'), mk('motionKernel'), mk('matchKernel'),
      ]);
      return new WebGpuAccelerator(device, grayPipeline, sobelPipeline, motionPipeline, matchPipeline);
    } catch (e) {
      console.warn('WebGPU 管线创建失败，回退 CPU：', e);
      return null;
    }
  }

  /** 一帧的灰度+梯度+运动掩码（首帧 prev 为零缓冲） */
  async processFrame(image: ImageData, prevGray: Float32Array | null): Promise<GpuFrameBuffers> {
    const { width: w, height: h } = image;
    const n = w * h;
    const device = this.device;
    const rgbaCopy = new Uint8Array(image.data);
    const rgbaBuf = this.makeBuffer(rgbaCopy.byteLength, GPUBufferUsage.STORAGE, rgbaCopy);
    const grayBuf = this.makeBuffer(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const gxBuf = this.makeBuffer(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const gyBuf = this.makeBuffer(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const magBuf = this.makeBuffer(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const prevBuf = this.makeBuffer(n * 4, GPUBufferUsage.STORAGE, prevGray ?? new Float32Array(n));
    const motionBuf = this.makeBuffer(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const dimBuf = this.makeBuffer(16, GPUBufferUsage.UNIFORM, new Uint32Array([w, h, 0, 0]));

    const encoder = device.createCommandEncoder();
    const groups: [number, number] = [Math.ceil(w / 16), Math.ceil(h / 16)];
    this.dispatch(encoder, this.grayPipeline, [dimBuf, rgbaBuf, grayBuf], groups);
    this.dispatch(encoder, this.sobelPipeline,
      [dimBuf, rgbaBuf, grayBuf, gxBuf, gyBuf, magBuf, prevBuf, motionBuf], groups);
    this.dispatch(encoder, this.motionPipeline,
      [dimBuf, rgbaBuf, grayBuf, gxBuf, gyBuf, magBuf, prevBuf, motionBuf], groups);

    const staged = [
      this.stageCopy(encoder, grayBuf, n * 4),
      this.stageCopy(encoder, gxBuf, n * 4),
      this.stageCopy(encoder, gyBuf, n * 4),
      this.stageCopy(encoder, magBuf, n * 4),
      this.stageCopy(encoder, motionBuf, n * 4),
    ];
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const [grayAb, gxAb, gyAb, magAb, motionAb] = await Promise.all(staged.map((s) => this.mapStaged(s)));
    const gray = new Float32Array(grayAb);
    const gx = new Float32Array(gxAb);
    const gy = new Float32Array(gyAb);
    const mag = new Float32Array(magAb);
    const motionU32 = new Uint32Array(motionAb);
    const motion = new Uint8Array(n);
    let count = 0;
    for (let i = 0; i < n; i++) {
      motion[i] = motionU32[i];
      count += motionU32[i];
    }
    return { gray, gx, gy, mag, motion, motionRatio: count / n };
  }

  /** 描述子暴力匹配，返回每个 A 的最佳 B 索引（-1 表示未通过比率测试） */
  async matchDescriptors(
    descA: Float32Array, countA: number, descB: Float32Array, countB: number,
    ratio = 0.78, maxDist = 0.9,
  ): Promise<{ index: Int32Array; distance: Float32Array }> {
    const device = this.device;
    const cfgBuf = this.makeBuffer(16, GPUBufferUsage.UNIFORM,
      new Float32Array([countA, countB, ratio, maxDist]));
    const aBuf = this.makeBuffer(descA.byteLength, GPUBufferUsage.STORAGE, descA);
    const bBuf = this.makeBuffer(descB.byteLength, GPUBufferUsage.STORAGE, descB);
    const idxBuf = this.makeBuffer(countA * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const distBuf = this.makeBuffer(countA * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const encoder = device.createCommandEncoder();
    this.dispatch(encoder, this.matchPipeline,
      [cfgBuf, aBuf, bBuf, idxBuf, distBuf], [Math.ceil(countA / 64), 1]);
    const staged = [
      this.stageCopy(encoder, idxBuf, countA * 4),
      this.stageCopy(encoder, distBuf, countA * 4),
    ];
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const [idxAb, distAb] = await Promise.all(staged.map((s) => this.mapStaged(s)));
    return { index: new Int32Array(idxAb), distance: new Float32Array(distAb) };
  }

  // ---------- 内部工具 ----------

  private makeBuffer(size: number, usage: number, data?: ArrayBufferView): GPUBuffer {
    const buf = this.device.createBuffer({
      size: Math.max(4, Math.ceil(size / 4) * 4),
      usage: usage | (data ? GPUBufferUsage.COPY_DST : 0),
    });
    if (data) this.device.queue.writeBuffer(buf, 0, data as BufferSource);
    return buf;
  }

  private dispatch(
    encoder: GPUCommandEncoder, pipeline: GPUComputePipeline,
    buffers: GPUBuffer[], groups: [number, number],
  ): void {
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups[0], groups[1]);
    pass.end();
  }

  /** 在 encoder 上登记一次到 MAP_READ 缓冲的拷贝，提交后用 mapStaged 取回 */
  private stageCopy(encoder: GPUCommandEncoder, src: GPUBuffer, size: number): GPUBuffer {
    const out = this.device.createBuffer({
      size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.copyBufferToBuffer(src, 0, out, 0, size);
    return out;
  }

  private async mapStaged(buf: GPUBuffer): Promise<ArrayBuffer> {
    await buf.mapAsync(GPUMapMode.READ);
    const copy = buf.getMappedRange().slice(0);
    buf.unmap();
    buf.destroy();
    return copy;
  }
}
