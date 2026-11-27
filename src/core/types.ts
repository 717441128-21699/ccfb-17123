// 竹林重建系统的核心领域类型

export type Vec2 = { x: number; y: number };
export type Vec3 = { x: number; y: number; z: number };

/** 竹竿弯曲：水平位置随高度 z 的二次偏移 p(z) = base + lean*z + bend*z^2 */
export interface BendModel {
  /** 地面处水平位置（米） */
  base: Vec2;
  /** 线性倾斜（米/米） */
  lean: Vec2;
  /** 二次弯曲系数（1/米） */
  bend: Vec2;
}

/** 真值里的一根竹竿（合成数据生成与评测用） */
export interface GroundTruthCulm {
  id: number;
  bend: BendModel;
  heightM: number;
  /** 胸径（米），1.3 m 高度处的外径 */
  diameterM: number;
  /** 竹节高度（米，自地面起，升序） */
  nodeHeightsM: number[];
  /** 竹龄（年，1~5） */
  ageYears: number;
  /** 竹竿基色（线性 RGB，0~1） */
  baseColor: Vec3;
  /** 竿面斑痕（归一化高度 u ∈ [0,1]，周向角 theta，半径比例） */
  spots: { u: number; theta: number; r: number; darkness: number }[];
}

/** 已知直径的标尺竿（师傅带进林子的红漆杆） */
export interface ReferencePole {
  bend: BendModel;
  heightM: number;
  /** 已知真实直径（米） */
  diameterM: number;
}

/** 一整片竹林的真值 */
export interface GroveGroundTruth {
  culms: GroundTruthCulm[];
  pole: ReferencePole;
  /** 每帧相机外参（世界 -> 相机）与内参 */
  frames: GroundTruthFrame[];
  /** 生成参数 */
  params: GroveGenParams;
}

export interface GroundTruthFrame {
  index: number;
  /** 相机位置（米） */
  position: Vec3;
  /** 注视点（米） */
  target: Vec3;
  up: Vec3;
  fovYDeg: number;
}

export type GroveDensity = 'sparse' | 'medium' | 'dense';
export type WindLevel = 'calm' | 'breeze' | 'windy';

export interface GroveGenParams {
  density: GroveDensity;
  wind: WindLevel;
  /** 竹林地块半径（米） */
  plotRadiusM: number;
  culmCount: number;
  frameCount: number;
  width: number;
  height: number;
  fovYDeg: number;
  /** 随机种子（同种子可复现） */
  seed: number;
  /** 环拍半径（米） */
  orbitRadiusM: number;
}

// ---------- 视觉管线中间产物 ----------

/** 一帧上检测到的竖向棱线段（像素） */
export interface VerticalEdge {
  /** 顶端像素坐标（x 为亚像素） */
  xTop: number;
  xBottom: number;
  yTop: number;
  yBottom: number;
  /** 棱强度 0~1 */
  strength: number;
}

/** 竹节环 / 斑痕特征点（跨帧匹配的锚点） */
export interface CulmFeature {
  /** 像素位置（亚像素） */
  px: number;
  py: number;
  /** 特征类型：竹节环上的点 / 竿面斑痕 */
  kind: 'node-ring' | 'spot';
  /** 描述子（SIFT 风格的 8 方向梯度直方图，已归一化） */
  descriptor: Float32Array;
  /** 所在竖棱索引（同帧内），-1 表示未归属 */
  edgeIndex: number;
  /** 特征尺度（像素） */
  scale: number;
}

export interface FrameObservations {
  index: number;
  edges: VerticalEdge[];
  features: CulmFeature[];
  /** 运动掩码占比 0~1（风动竹叶区域被剔除的比例） */
  motionMaskRatio: number;
  /** 天空/背景掩码占比 */
  skyMaskRatio: number;
  /** 曝光时间戳（秒，来自视频） */
  timeSec: number;
}

/** 特征匹配对 */
export interface FeatureMatch {
  aFrame: number;
  aFeature: number;
  bFrame: number;
  bFeature: number;
  /** Lowe 比率后的距离，越小越好 */
  distance: number;
}

/** 相机位姿（世界 -> 相机的旋转平移），尺度为估计值，标尺竿校正后变米制 */
export interface CameraPose {
  index: number;
  /** 世界->相机旋转，行主序 3x3 */
  R: Float32Array;
  t: Vec3;
  position: Vec3;
  /** 米制尺度（sim(3) 的 s，闭环校正后为 1） */
  scale: number;
}

/** 重建出的竹竿 */
export interface ReconstructedCulm {
  id: number;
  bend: BendModel;
  /** 胸径估计（米） */
  diameterM: number;
  diameterSigmaM: number;
  /** 竹节高度（米） */
  nodeHeightsM: number[];
  /** 竹龄估计（年，浮点）与置信度 0~1 */
  ageYears: number;
  ageConfidence: number;
  /** 平均竿色（线性 RGB） */
  meanColor: Vec3;
  /** 被多少帧观测到 */
  observationCount: number;
  /** 拟合残差（像素） */
  reprojectionErrorPx: number;
}

/** 重建总结果 */
export interface ReconstructionResult {
  culms: ReconstructedCulm[];
  poses: CameraPose[];
  /** 标尺竿在重建中的直径（米），用于尺度校正 */
  poleDiameterEstimateM: number;
  poleDiameterTruthM: number;
  /** 尺度因子：真值米 = 估计值 * scaleCorrection */
  scaleCorrection: number;
  timings: PipelineTimings;
}

export interface PipelineTimings {
  frameExtractMs: number;
  gpuFeatureMs: number;
  matchingMs: number;
  poseMs: number;
  cylinderFitMs: number;
  nodeMs: number;
  ageMs: number;
  totalMs: number;
}

// ---------- 选竹规格 ----------

/** 一个笛子调门的用料要求（直径为胸径外径，节长为相邻竹节间距） */
export interface FluteSpec {
  key: string;
  label: string;
  /** 目标胸径（毫米） */
  diameterMm: number;
  /** 胸径容差（毫米） */
  diameterTolMm: number;
  /** 最短可用节长（毫米，笛身两段长孔距决定） */
  minNodeSpacingMm: number;
  /** 最低竹龄（年） */
  minAge: number;
}

/** 选竹标注结果 */
export interface BambooSelection {
  culmId: number;
  specKey: string;
  /** 综合适配评分 0~1 */
  score: number;
  reasons: string[];
}

// ---------- 评测 ----------

export interface AccuracyReport {
  /** 位置 RMSE（米） */
  positionRmseM: number;
  /** 胸径 RMSE（毫米） */
  diameterRmseMm: number;
  /** 竹节间距 RMSE（毫米） */
  nodeSpacingRmseMm: number;
  /** 一根被拆成两根的次数 */
  splitErrors: number;
  /** 两根被并成一根的次数 */
  mergeErrors: number;
  /** 正确匹配的重建竹竿数 */
  matchedCulms: number;
  truthCount: number;
  reconCount: number;
  /** 竹龄分类准确率（±1 年视为正确） */
  ageAccuracy: number;
}
