// 合成竹林渲染的 GLSL ES 3.00 着色器
// 场景约定：世界坐标 y 向上，竹竿沿 y 生长，弯曲轴 p(y)=base+lean*y+bend*y^2

const FULL_HEADER = `#version 300 es
precision highp float;
`;

// ---------- 竹竿：索引化实例圆柱，弯曲在顶点着色器中计算，竹节走 uniform（逐实例设置） ----------

export const CULM_VERT = `${FULL_HEADER}
// 模板圆柱顶点（局部坐标，y ∈ [0,1]，径向在 xz 单位圆）
in vec3 aTemplate;
// 每实例属性（location 1..3）
in vec4 iBaseHeightRadius; // loc1: base.xy, height, radius
in vec4 iLeanBend;         // loc2: lean.xy, bend.xy
in vec4 iColorAge;         // loc3: 基色 rgb，竹龄

uniform mat4 uViewProj;

out vec3 vWorld;
out float vHeightN;
out float vTheta;
out vec3 vNormalW;
out vec3 vColor;
out float vAge;

void main() {
  vec2 base = iBaseHeightRadius.xy;
  float h = iBaseHeightRadius.z;
  float radius = iBaseHeightRadius.w;
  vec2 lean = iLeanBend.xy;
  vec2 bend = iLeanBend.zw;
  float yN = aTemplate.y;
  float y = yN * h;
  vec2 axis = base + lean * y + bend * y * y;
  vec2 axisD = lean + 2.0 * bend * y;
  vec3 radial = normalize(vec3(aTemplate.x, 0.0, aTemplate.z));
  vec3 pos = vec3(axis.x + radial.x * radius, y, axis.y + radial.z * radius);
  vec3 normalW = normalize(radial - vec3(axisD.x, 0.0, axisD.y) * radial.y);
  vWorld = pos;
  vHeightN = yN;
  vTheta = atan(aTemplate.z, aTemplate.x);
  vNormalW = normalW;
  vColor = iColorAge.rgb;
  vAge = iColorAge.a;
  gl_Position = uViewProj * vec4(pos, 1.0);
}`;

export const CULM_FRAG = `${FULL_HEADER}
in vec3 vWorld;
in float vHeightN;
in float vTheta;
in vec3 vNormalW;
in vec3 vColor;
in float vAge;

uniform vec3 uCameraPos;
uniform float uNodeN[48];
uniform int uNodeCount;
out vec4 outColor;

float hash11(float n) { return fract(sin(n) * 43758.5453123); }

void main() {
  vec3 N = normalize(vNormalW);
  vec3 L = normalize(vec3(0.55, 0.8, 0.35));
  float diff = clamp(dot(N, L) * 0.5 + 0.5, 0.0, 1.0);
  vec3 V = normalize(uCameraPos - vWorld);
  float rim = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.5) * 0.18;

  vec3 col = vColor * (0.35 + 0.65 * diff) + rim * vec3(0.9, 1.0, 0.85);

  // 竹节环：深色窄环
  float nodeBand = 0.0;
  for (int i = 0; i < 48; i++) {
    if (i >= uNodeCount) break;
    float d = abs(vHeightN - uNodeN[i]);
    nodeBand = max(nodeBand, smoothstep(0.006, 0.0, d));
  }
  col = mix(col, col * 0.45, nodeBand * 0.85);

  // 斑痕：按世界位置稳定哈希的深色椭圆斑（帧间不漂移，可作匹配锚点）
  float spot = 0.0;
  for (int k = 0; k < 9; k++) {
    float seed = floor(vWorld.x * 13.0 + vWorld.z * 7.0) * 17.0 + float(k) * 3.1;
    float su = hash11(seed);
    float st = hash11(seed + 1.7) * 6.2831;
    float sr = 0.012 + hash11(seed + 2.3) * 0.02;
    float ang = mod(vTheta - st + 3.14159, 6.2831) - 3.14159;
    float du = vHeightN - su;
    float dt = ang * 0.04;
    float d = sqrt(du * du + dt * dt);
    spot = max(spot, (1.0 - smoothstep(sr, sr * 1.6, d)) * (0.3 + 0.4 * hash11(seed + 4.1)));
  }
  col = mix(col, col * 0.35, spot);

  // 新竹白粉：随竹龄减少的灰白色覆膜
  float powder = clamp(0.6 - 0.115 * vAge, 0.0, 0.55);
  float powderNoise = hash11(floor(vHeightN * 90.0) + floor(vTheta * 20.0) * 7.0);
  col = mix(col, vec3(0.82, 0.86, 0.78), powder * (0.5 + 0.5 * powderNoise));

  // 距离雾
  float dist = length(uCameraPos - vWorld);
  float fog = smoothstep(16.0, 30.0, dist);
  col = mix(col, vec3(0.72, 0.78, 0.7), fog * 0.7);

  outColor = vec4(col, 1.0);
}`;

// ---------- 叶片：相机朝向 billboard，风动 ----------

export const LEAF_VERT = `${FULL_HEADER}
in vec3 aTemplate;
in vec4 iCenterScale; // 簇心 xyz，尺度
in vec4 iPhaseColor;  // 相位，叶色 rgb

uniform mat4 uViewProj;
uniform float uTime;
uniform float uWindAmp;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;

out vec2 vUv;
out vec3 vColor;
out float vPhase;

void main() {
  vec3 center = iCenterScale.xyz;
  float s = iCenterScale.w;
  float phase = iPhaseColor.x;
  float gust = sin(uTime * 2.3 + phase) + 0.5 * sin(uTime * 4.7 + phase * 1.7);
  vec3 sway = vec3(gust * uWindAmp * (0.4 + center.y * 0.12), 0.0,
                   gust * uWindAmp * 0.5 * (0.4 + center.y * 0.1));
  vec3 c = center + sway;
  vec3 pos = c + (uCameraRight * aTemplate.x + uCameraUp * aTemplate.y) * s;
  vUv = aTemplate.xy;
  vColor = iPhaseColor.yzw;
  vPhase = phase;
  gl_Position = uViewProj * vec4(pos, 1.0);
}`;

export const LEAF_FRAG = `${FULL_HEADER}
in vec2 vUv;
in vec3 vColor;
in float vPhase;
out vec4 outColor;

float hash11(float n) { return fract(sin(n) * 43758.5453123); }

void main() {
  vec2 q = vec2(vUv.x, vUv.y * 0.45);
  float d = abs(q.x) + abs(q.y);
  float alpha = 1.0 - smoothstep(0.75, 1.0, d);
  if (alpha < 0.08) discard;
  float vein = smoothstep(0.06, 0.0, abs(q.x)) * 0.18;
  float jitter = hash11(floor((vUv.x + 2.0) * 8.0) + floor((vUv.y + 2.0) * 8.0) + vPhase) * 0.12;
  vec3 col = vColor * (0.75 + jitter) + vein;
  outColor = vec4(col, alpha);
}`;

// ---------- 地面 ----------

export const GROUND_VERT = `${FULL_HEADER}
in vec2 aPos;
uniform mat4 uViewProj;
out vec3 vWorld;
void main() {
  vWorld = vec3(aPos.x, 0.0, aPos.y);
  gl_Position = uViewProj * vec4(vWorld, 1.0);
}`;

export const GROUND_FRAG = `${FULL_HEADER}
in vec3 vWorld;
uniform vec3 uCameraPos;
out vec4 outColor;
float hash11(float n) { return fract(sin(n) * 43758.5453123); }
void main() {
  float g = hash11(floor(vWorld.x * 2.0) * 13.0 + floor(vWorld.z * 2.0) * 7.0);
  vec3 col = mix(vec3(0.28, 0.34, 0.2), vec3(0.36, 0.4, 0.24), g);
  float litter = hash11(floor(vWorld.x * 6.0) * 5.0 + floor(vWorld.z * 6.0) * 11.0);
  col = mix(col, vec3(0.5, 0.42, 0.22), step(0.93, litter) * 0.6);
  float dist = length(uCameraPos.xz - vWorld.xz);
  float fog = smoothstep(18.0, 32.0, dist);
  col = mix(col, vec3(0.72, 0.78, 0.7), fog * 0.7);
  outColor = vec4(col, 1.0);
}`;

// ---------- 红漆标尺竿（无竹节，单实例） ----------

export const POLE_VERT = `${FULL_HEADER}
in vec3 aTemplate;
in vec4 iBaseHeightRadius;
in vec4 iLeanBend;
uniform mat4 uViewProj;
out vec3 vWorld;
out float vHeightN;
out vec3 vNormalW;
void main() {
  vec2 base = iBaseHeightRadius.xy;
  float h = iBaseHeightRadius.z;
  float radius = iBaseHeightRadius.w;
  vec2 lean = iLeanBend.xy;
  vec2 bend = iLeanBend.zw;
  float yN = aTemplate.y;
  float y = yN * h;
  vec2 axis = base + lean * y + bend * y * y;
  vec2 axisD = lean + 2.0 * bend * y;
  vec3 radial = normalize(vec3(aTemplate.x, 0.0, aTemplate.z));
  vec3 pos = vec3(axis.x + radial.x * radius, y, axis.y + radial.z * radius);
  vNormalW = normalize(radial - vec3(axisD.x, 0.0, axisD.y) * radial.y);
  vWorld = pos;
  vHeightN = yN;
  gl_Position = uViewProj * vec4(pos, 1.0);
}`;

export const POLE_FRAG = `${FULL_HEADER}
in vec3 vWorld;
in float vHeightN;
in vec3 vNormalW;
uniform vec3 uCameraPos;
out vec4 outColor;
void main() {
  vec3 N = normalize(vNormalW);
  vec3 L = normalize(vec3(0.55, 0.8, 0.35));
  float diff = clamp(dot(N, L) * 0.5 + 0.5, 0.0, 1.0);
  float seg = mod(floor(vHeightN * 10.0), 2.0);
  vec3 base = mix(vec3(0.85, 0.12, 0.1), vec3(0.92, 0.9, 0.86), seg);
  outColor = vec4(base * (0.35 + 0.65 * diff), 1.0);
}`;
