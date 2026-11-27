// WebGL2 合成竹林渲染器：索引化实例竹竿/叶片/标尺竿，输出帧供视频编码与视觉管线使用
import {
  mat4LookAt, mat4Mul, mat4Perspective, v3,
} from '../core/linalg';
import { Rng } from '../core/rng';
import { bendAxisAt } from '../core/grove';
import type { GroveGroundTruth, GroundTruthFrame, Vec3 } from '../core/types';
import {
  CULM_FRAG, CULM_VERT, GROUND_FRAG, GROUND_VERT,
  LEAF_FRAG, LEAF_VERT, POLE_FRAG, POLE_VERT,
} from './shaders';

const MAX_NODES = 48;
const CULM_INSTANCE_VEC4S = 3; // base/lean/color 三个 vec4，竹节走 uniform
const CULM_INSTANCE_STRIDE = CULM_INSTANCE_VEC4S * 16;
const RADIAL_SEGMENTS = 18;
const HEIGHT_RINGS = 26;
const COLS = RADIAL_SEGMENTS + 1;
const LEAF_CLUSTERS_PER_CULM = 5;
const LEAVES_PER_CLUSTER = 5;

const WIND_AMPS: Record<string, number> = { calm: 0.0, breeze: 0.06, windy: 0.18 };

interface GlProgram {
  prog: WebGLProgram;
  attribs: Record<string, number>;
  uniforms: Record<string, WebGLUniformLocation | null>;
}

export class GroveRenderer {
  private gl: WebGL2RenderingContext;
  private culmProg: GlProgram;
  private leafProg: GlProgram;
  private groundProg: GlProgram;
  private poleProg: GlProgram;
  private culmVao: WebGLVertexArrayObject;
  private leafVao: WebGLVertexArrayObject;
  private groundVao: WebGLVertexArrayObject;
  private poleVao: WebGLVertexArrayObject;
  private culmInstanceBuf: WebGLBuffer;
  private leafInstanceBuf: WebGLBuffer;
  private poleInstanceBuf: WebGLBuffer;
  private templateCulmBuf: WebGLBuffer;
  private templateLeafBuf: WebGLBuffer;
  private groundBuf: WebGLBuffer;
  private indexBuf: WebGLBuffer;
  private indexCount = 0;
  private culmCount = 0;
  private leafCount = 0;

  constructor(private canvas: HTMLCanvasElement, private gt: GroveGroundTruth) {
    const gl = canvas.getContext('webgl2', {
      antialias: true, preserveDrawingBuffer: false, alpha: false,
    });
    if (!gl) throw new Error('当前浏览器不支持 WebGL2，无法渲染合成竹林');
    this.gl = gl;
    const culmAttribs = ['aTemplate', 'iBaseHeightRadius', 'iLeanBend', 'iColorAge'];
    this.culmProg = this.buildProgram(CULM_VERT, CULM_FRAG, culmAttribs,
      ['uViewProj', 'uCameraPos', 'uNodeN', 'uNodeCount']);
    this.leafProg = this.buildProgram(LEAF_VERT, LEAF_FRAG,
      ['aTemplate', 'iCenterScale', 'iPhaseColor'],
      ['uViewProj', 'uTime', 'uWindAmp', 'uCameraRight', 'uCameraUp']);
    this.groundProg = this.buildProgram(GROUND_VERT, GROUND_FRAG, ['aPos'],
      ['uViewProj', 'uCameraPos']);
    this.poleProg = this.buildProgram(POLE_VERT, POLE_FRAG,
      ['aTemplate', 'iBaseHeightRadius', 'iLeanBend'], ['uViewProj', 'uCameraPos']);

    this.templateCulmBuf = this.createBuffer(this.buildCulmTemplate());
    this.templateLeafBuf = this.createBuffer(this.buildLeafTemplate());
    this.groundBuf = this.createBuffer(this.buildGround());
    this.indexBuf = this.createIndexBuffer();
    this.culmInstanceBuf = this.createBuffer(null);
    this.leafInstanceBuf = this.createBuffer(null);
    this.poleInstanceBuf = this.createBuffer(null);
    this.culmVao = this.buildCulmVao();
    this.leafVao = this.buildLeafVao();
    this.groundVao = this.buildGroundVao();
    this.poleVao = this.buildPoleVao();
    this.uploadCulms();
    this.uploadLeaves();
    this.uploadPole();
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.72, 0.78, 0.7, 1);
  }

  get windAmp(): number {
    return WIND_AMPS[this.gt.params.wind] ?? 0;
  }

  /** 渲染第 frameIndex 帧；timeSec 控制风动相位 */
  renderFrame(frameIndex: number, timeSec: number): void {
    const gl = this.gl;
    const f = this.gt.frames[frameIndex];
    this.canvas.width = this.gt.params.width;
    this.canvas.height = this.gt.params.height;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const aspect = this.canvas.width / this.canvas.height;
    const proj = mat4Perspective((f.fovYDeg * Math.PI) / 180, aspect, 0.1, 120);
    const view = mat4LookAt(f.position, f.target, f.up);
    const vp = mat4Mul(proj, view);
    const { right, up } = frameCameraVectors(f);

    this.drawGround(vp, f.position);
    this.drawLeaves(vp, f.position, right, up, timeSec);
    this.drawCulms(vp, f.position);
    this.drawPole(vp, f.position);
  }

  /** 读取当前帧像素（RGBA，左上原点） */
  readPixels(): Uint8ClampedArray {
    const gl = this.gl;
    const { width, height } = this.canvas;
    const raw = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, raw);
    const out = new Uint8ClampedArray(width * height * 4);
    const row = width * 4;
    for (let y = 0; y < height; y++) {
      out.set(raw.subarray((height - 1 - y) * row, (height - y) * row), y * row);
    }
    return out;
  }

  // ---------- 几何 ----------

  private buildCulmTemplate(): Float32Array {
    const verts: number[] = [];
    for (let j = 0; j <= HEIGHT_RINGS; j++) {
      const y = j / HEIGHT_RINGS;
      for (let i = 0; i <= RADIAL_SEGMENTS; i++) {
        const a = (i / RADIAL_SEGMENTS) * Math.PI * 2;
        verts.push(Math.cos(a), y, Math.sin(a));
      }
    }
    return Float32Array.from(verts);
  }

  private buildLeafTemplate(): Float32Array {
    return Float32Array.of(-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0);
  }

  private buildGround(): Float32Array {
    const r = this.gt.params.plotRadiusM * 2.2;
    return Float32Array.of(-r, -r, r, -r, r, r, -r, -r, r, r, -r, r);
  }

  private createIndexBuffer(): WebGLBuffer {
    const indices: number[] = [];
    for (let j = 0; j < HEIGHT_RINGS; j++) {
      for (let i = 0; i < RADIAL_SEGMENTS; i++) {
        const a = j * COLS + i;
        const b = a + 1;
        const c = a + COLS;
        const d = c + 1;
        indices.push(a, b, c, b, d, c);
      }
    }
    this.indexCount = indices.length;
    const gl = this.gl;
    const buf = gl.createBuffer();
    if (!buf) throw new Error('索引 Buffer 创建失败');
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint16Array.from(indices), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    return buf;
  }

  // ---------- 实例数据 ----------

  private nodeNormByCulm: Float32Array[] = [];
  private nodeCountByCulm: number[] = [];

  private uploadCulms(): void {
    const gl = this.gl;
    const data = new Float32Array(this.gt.culms.length * CULM_INSTANCE_VEC4S * 4);
    this.nodeNormByCulm = [];
    this.nodeCountByCulm = [];
    this.gt.culms.forEach((c, idx) => {
      const b = c.bend;
      const o = idx * CULM_INSTANCE_VEC4S * 4;
      data.set([b.base.x, b.base.y, c.heightM, c.diameterM / 2], o);
      data.set([b.lean.x, b.lean.y, b.bend.x, b.bend.y], o + 4);
      data.set([c.baseColor.x, c.baseColor.y, c.baseColor.z, c.ageYears], o + 8);
      const nodeNorm = c.nodeHeightsM.slice(0, MAX_NODES).map((z) => z / c.heightM);
      const padded = new Float32Array(MAX_NODES).fill(-1);
      padded.set(nodeNorm);
      this.nodeNormByCulm.push(padded);
      this.nodeCountByCulm.push(nodeNorm.length);
    });
    this.culmCount = this.gt.culms.length;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.culmInstanceBuf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  }

  private uploadLeaves(): void {
    const gl = this.gl;
    const rng = new Rng(this.gt.params.seed ^ 0x5eed);
    const data: number[] = [];
    for (const c of this.gt.culms) {
      for (let k = 0; k < LEAF_CLUSTERS_PER_CULM; k++) {
        const zN = 0.55 + rng.range(0, 0.4);
        const z = zN * c.heightM;
        const axis = bendAxisAt(c.bend, z);
        const ang = rng.range(0, Math.PI * 2);
        const reach = c.diameterM * 2 + rng.range(0.1, 0.5);
        const cx = axis.x + Math.cos(ang) * reach;
        const cz = axis.y + Math.sin(ang) * reach;
        for (let j = 0; j < LEAVES_PER_CLUSTER; j++) {
          const s = rng.range(0.12, 0.26);
          const phase = rng.range(0, Math.PI * 2);
          data.push(
            cx + rng.range(-0.15, 0.15), c.heightM * zN + rng.range(-0.2, 0.2),
            cz + rng.range(-0.15, 0.15), s,
            phase, 0.18 + rng.range(0, 0.1), 0.32 + rng.range(0, 0.18), 0.1 + rng.range(0, 0.08),
          );
        }
      }
    }
    this.leafCount = data.length / 8;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.leafInstanceBuf);
    gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from(data), gl.STATIC_DRAW);
  }

  private uploadPole(): void {
    const gl = this.gl;
    const p = this.gt.pole;
    const b = p.bend;
    const data = Float32Array.of(
      b.base.x, b.base.y, p.heightM, p.diameterM / 2,
      b.lean.x, b.lean.y, b.bend.x, b.bend.y,
    );
    gl.bindBuffer(gl.ARRAY_BUFFER, this.poleInstanceBuf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  }

  // ---------- VAO ----------

  private buildCulmVao(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = this.requireVao();
    gl.bindVertexArray(vao);
    this.bindTemplateAttrib(this.templateCulmBuf, this.culmProg.attribs.aTemplate, 3);
    this.bindCulmInstanceAttribs(this.culmProg);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuf);
    gl.bindVertexArray(null);
    return vao;
  }

  private buildLeafVao(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = this.requireVao();
    gl.bindVertexArray(vao);
    this.bindTemplateAttrib(this.templateLeafBuf, this.leafProg.attribs.aTemplate, 3);
    const loc1 = this.leafProg.attribs.iCenterScale;
    const loc2 = this.leafProg.attribs.iPhaseColor;
    this.setInstanceAttrib(this.leafInstanceBuf, loc1, 4, 0, 32);
    this.setInstanceAttrib(this.leafInstanceBuf, loc2, 4, 16, 32);
    gl.bindVertexArray(null);
    return vao;
  }

  private buildGroundVao(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = this.requireVao();
    gl.bindVertexArray(vao);
    this.bindTemplateAttrib(this.groundBuf, this.groundProg.attribs.aPos, 2);
    gl.bindVertexArray(null);
    return vao;
  }

  private buildPoleVao(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = this.requireVao();
    gl.bindVertexArray(vao);
    this.bindTemplateAttrib(this.templateCulmBuf, this.poleProg.attribs.aTemplate, 3);
    this.setInstanceAttrib(this.poleInstanceBuf, this.poleProg.attribs.iBaseHeightRadius, 4, 0, 32);
    this.setInstanceAttrib(this.poleInstanceBuf, this.poleProg.attribs.iLeanBend, 4, 16, 32);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuf);
    gl.bindVertexArray(null);
    return vao;
  }

  private bindCulmInstanceAttribs(prog: GlProgram): void {
    const names = ['iBaseHeightRadius', 'iLeanBend', 'iColorAge'];
    names.forEach((name, i) => {
      this.setInstanceAttrib(this.culmInstanceBuf, prog.attribs[name], 4, i * 16, CULM_INSTANCE_STRIDE);
    });
  }

  // ---------- 绘制 ----------

  private drawCulms(vp: Float32Array, cam: Vec3): void {
    const gl = this.gl;
    gl.useProgram(this.culmProg.prog);
    gl.uniformMatrix4fv(this.culmProg.uniforms.uViewProj, false, vp);
    gl.uniform3f(this.culmProg.uniforms.uCameraPos, cam.x, cam.y, cam.z);
    // 竹节是逐实例 uniform：逐竿绘制，用 ARRAY_BUFFER 偏移选择该竿的实例数据
    const attribs = [
      this.culmProg.attribs.iBaseHeightRadius,
      this.culmProg.attribs.iLeanBend,
      this.culmProg.attribs.iColorAge,
    ];
    gl.bindVertexArray(this.culmVao);
    for (let i = 0; i < this.culmCount; i++) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.culmInstanceBuf);
      attribs.forEach((loc, k) => {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, CULM_INSTANCE_STRIDE, i * CULM_INSTANCE_STRIDE + k * 16);
        gl.vertexAttribDivisor(loc, 1);
      });
      gl.uniform1fv(this.culmProg.uniforms.uNodeN, this.nodeNormByCulm[i]);
      gl.uniform1i(this.culmProg.uniforms.uNodeCount, this.nodeCountByCulm[i]);
      gl.drawElementsInstanced(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0, 1);
    }
    gl.bindVertexArray(null);
  }

  private drawPole(vp: Float32Array, cam: Vec3): void {
    const gl = this.gl;
    gl.useProgram(this.poleProg.prog);
    gl.uniformMatrix4fv(this.poleProg.uniforms.uViewProj, false, vp);
    gl.uniform3f(this.poleProg.uniforms.uCameraPos, cam.x, cam.y, cam.z);
    gl.bindVertexArray(this.poleVao);
    gl.drawElementsInstanced(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0, 1);
    gl.bindVertexArray(null);
  }

  private drawLeaves(vp: Float32Array, _cam: Vec3, camRight: Vec3, camUp: Vec3, time: number): void {
    const gl = this.gl;
    const prog = this.leafProg;
    gl.useProgram(prog.prog);
    gl.uniformMatrix4fv(prog.uniforms.uViewProj, false, vp);
    gl.uniform1f(prog.uniforms.uTime, time);
    gl.uniform1f(prog.uniforms.uWindAmp, this.windAmp);
    gl.uniform3f(prog.uniforms.uCameraRight, camRight.x, camRight.y, camRight.z);
    gl.uniform3f(prog.uniforms.uCameraUp, camUp.x, camUp.y, camUp.z);
    gl.bindVertexArray(this.leafVao);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.leafCount);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  private drawGround(vp: Float32Array, cam: Vec3): void {
    const gl = this.gl;
    gl.useProgram(this.groundProg.prog);
    gl.uniformMatrix4fv(this.groundProg.uniforms.uViewProj, false, vp);
    gl.uniform3f(this.groundProg.uniforms.uCameraPos, cam.x, cam.y, cam.z);
    gl.bindVertexArray(this.groundVao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  // ---------- GL 辅助 ----------

  private requireVao(): WebGLVertexArrayObject {
    const vao = this.gl.createVertexArray();
    if (!vao) throw new Error('VAO 创建失败');
    return vao;
  }

  private bindTemplateAttrib(buf: WebGLBuffer, loc: number, size: number): void {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }

  private setInstanceAttrib(buf: WebGLBuffer, loc: number, size: number, offset: number, stride: number): void {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
    gl.vertexAttribDivisor(loc, 1);
  }

  private createBuffer(data: Float32Array | null): WebGLBuffer {
    const gl = this.gl;
    const buf = gl.createBuffer();
    if (!buf) throw new Error('Buffer 创建失败');
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    if (data) gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return buf;
  }

  private buildProgram(
    vsSrc: string, fsSrc: string, attribNames: string[], uniformNames: string[],
  ): GlProgram {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type);
      if (!sh) throw new Error('Shader 创建失败');
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error(`着色器编译失败: ${log}\n${src}`);
      }
      return sh;
    };
    const prog = gl.createProgram();
    if (!prog) throw new Error('Program 创建失败');
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`Program 链接失败: ${gl.getProgramInfoLog(prog)}`);
    }
    const attribs: Record<string, number> = {};
    for (const n of attribNames) attribs[n] = gl.getAttribLocation(prog, n);
    const uniforms: Record<string, WebGLUniformLocation | null> = {};
    for (const n of uniformNames) uniforms[n] = gl.getUniformLocation(prog, n);
    return { prog, attribs, uniforms };
  }
}

/** 计算某帧相机右/上方向 */
export function frameCameraVectors(f: GroundTruthFrame) {
  const d = { x: f.target.x - f.position.x, y: 0, z: f.target.z - f.position.z };
  const n = Math.hypot(d.x, d.z) || 1;
  const fwd = { x: d.x / n, y: 0, z: d.z / n };
  return { right: { x: -fwd.z, y: 0, z: fwd.x }, up: v3(0, 1, 0) };
}
