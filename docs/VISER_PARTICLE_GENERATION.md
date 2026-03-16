# Viser 粒子生成完整流程

## 📍 終端位置：瀏覽器中看到的粒子

用戶在 `http://0.0.0.0:8080` 看到的每個粒子都來自於：

```
粒子 = (中心位置, RGB 顏色, 大小)
      ↓        ↓         ↓
    means    colors    point_size
```

---

## 🔬 粒子的完整數據來源

### 第 1 層：DA3 模型預測 (GPU 上的 PyTorch Tensors)

**位置**: `src/depth_anything_3/specs.py` → Gaussians 數據結構

```python
@dataclass
class Gaussians:
    """3DGS parameters, all in world space (世界座標)"""
    
    means: torch.Tensor        # 世界座標粒子中心 (batch, N, 3) float32
    scales: torch.Tensor       # 縮放係數 (batch, N, 3) float32
    rotations: torch.Tensor    # 四元數 wxyz (batch, N, 4) float32
    harmonics: torch.Tensor    # 球諧係數 SH (batch, N, 3, d_sh) float32
    opacities: torch.Tensor    # 不透明度 (batch, N,) 或 (batch, N, 1, d_sh) float32
```

### 第 2 層：GaussianAdapter 轉換

**位置**: `src/depth_anything_3/model/gs_adapter.py` → `GaussianAdapter.forward()`

**輸入**：
```python
extrinsics: (batch, view, 4, 4)      # 相機外參 (世界→相機)
intrinsics: (batch, view, 3, 3)      # 相機內參
depths: (batch, view, H, W)          # 深度圖
opacities_prob: (batch, view, H, W)  # 不透明度概率
raw_gaussians: (batch, view, H, W, D) # 原始高斯特徵
```

**轉換步驟**：

#### 2.1 轉換到世界座標

```python
# 1) 計算相機到世界的變換
cam2worlds = affine_inverse(extrinsics)  # (batch, view, 4, 4)

# 2) 反投影深度 + XY 偏移到世界射線
# raw_gaussians[:, :, :, :, :2] 是 XY 偏移
# raw_gaussians[:, :, :, :, 2:] 包含深度、縮放、旋轉、SH

# 3) 計算世界空間位置
origins, directions = get_world_rays(...)  # 從相機內參計算射線
gs_means_world = origins + directions * gs_depths[..., None]
# → gs_means_world: (batch, N, 3) 世界座標 ✅
```

#### 2.2 計算縮放（考慮深度）

```python
# 縮放考慮深度變化（越遠的物體高斯越大以覆蓋更多像素）
scales = scale_min + (scale_max - scale_min) * raw_scales.sigmoid()
pixel_size = 1 / torch.tensor((W, H))  # 縮放到圖像空間
multiplier = get_scale_multiplier(intr_normed, pixel_size)

gs_scales = scales * gs_depths[..., None] * multiplier[..., None, None, None]
# → gs_scales: (batch, N, 3) ✅
```

#### 2.3 計算旋轉（世界空間）

```python
# 標準化四元數
rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

# 如果需要，旋轉到世界座標系
rotations_world = rotate_quaternions(rotations, cam2worlds[:, :, :3, :3])
# → rotations: (batch, N, 4) wxyz 四元數 ✅
```

#### 2.4 計算 SH（球諧係數）

```python
sh = raw_gaussians[..., end_of_scales_rots:]  # 提取 SH 部分

if sh_needs_rotation:
    # 旋轉 SH 到世界座標系
    gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
else:
    gs_sh_world = sh

# → harmonics: (batch, N, 3, d_sh) ✅
```

#### 2.5 計算不透明度

```python
# 從概率密度映射到不透明度
opacities = map_pdf_to_opacity(densities)
# → opacities: (batch, N,) ✅

return Gaussians(
    means=gs_means_world,           # (B, N, 3)
    scales=gs_scales,               # (B, N, 3)
    rotations=gs_rotations_world,   # (B, N, 4)
    harmonics=gs_sh_world,          # (B, N, 3, d_sh)
    opacities=gs_opacities          # (B, N,)
)
```

### 第 3 層：推理輸出 (InferenceEngine)

**位置**: `src/pipeline/inference_engine.py` → `_infer_pytorch()`

```python
output = self._model.model(
    batch,              # (1, 2, 3, H', W') 預處理圖像
    extrinsics=None,    # camera_params_mode="auto" 時為 None
    intrinsics=None     # 模型自行估計相機參數
)

# 模型返回
output.gaussians  # ⭐ Gaussians object (GPU 張量)
```

**狀態**：
- **位置**: GPU (CUDA device)
- **類型**: Gaussians 對象，所有字段都是 torch.Tensor
- **幀數**: batch=1, per-frame gaussian 數量 N （取決於深度圖解析度）

### 第 4 層：深度解碼器 (DepthDecoder)

**位置**: `src/pipeline/depth_decoder.py` → `decode()`

**操作**：
```python
def decode(self, inference_result):
    # 提取當前時刻幀
    gaussians = inference_result.get("gaussians", None)
    
    # 保留原樣，傳遞給渲染線程
    return {
        ...
        'gaussians': gaussians,  # ⭐ 仍在 GPU 上
        ...
    }
```

**狀態**：
- Gaussians 對象保持不變
- 仍在 GPU 上
- 所有 means, scales, rotations, harmonics, opacities 仍為 torch.Tensor

### 第 5 層：GPU→NumPy 轉換 (RenderThread)

**位置**: `src/realtime_pipeline.py` → `_process_da3_gaussians()`

**轉換矩陣**：

| 轉換階段 | 輸入 | 操作 | 輸出 | 說明 |
|---------|------|------|------|------|
| 1. GPU 張量 | `gaussians.means` | `detach()` | 斷開計算圖 | 準備轉換 |
| 2. CPU 移動 | `.detach()` | `.cpu()` | CPU 張量 | 從 GPU 複製 |
| 3. NumPy 轉換 | CPU 張量 | `.numpy()` | NumPy 數組 | 可渲染格式 |
| 4. 顏色生成 | `harmonics` | 提取 SH DC 項 | RGB 顏色 | `[0, 255]` uint8 |

**詳細代碼**：

```python
def _process_da3_gaussians(gaussians_obj):
    if gaussians_obj is None:
        return None
    
    result = {}
    
    # ✅ 第一步：提取位置（必需）
    means = gaussians_obj.means  # (N, 3) torch [GPU]
    result['means'] = means.detach().cpu().numpy()  # (N, 3) numpy float32 [CPU]
    
    # ✅ 第二步：提取縮放
    scales = gaussians_obj.scales  # (N, 3) torch [GPU]
    result['scales'] = scales.detach().cpu().numpy()  # (N, 3) numpy float32 [CPU]
    
    # ✅ 第三步：提取旋轉
    rotations = gaussians_obj.rotations  # (N, 4) torch [GPU]
    result['rotations'] = rotations.detach().cpu().numpy()  # (N, 4) numpy [CPU]
    
    # ✅ 第四步：提取 SH 係數
    harmonics = gaussians_obj.harmonics  # (B, N, 3, d_sh) torch [GPU]
    result['harmonics'] = harmonics.detach().cpu().numpy()  # numpy [CPU]
    
    # ✅ 第五步：提取不透明度
    opacities = gaussians_obj.opacities  # (N,) torch [GPU]
    result['opacities'] = opacities.detach().cpu().numpy()  # (N,) numpy [CPU]
    
    # ✅ 第六步：從 SH 生成顏色
    means_np = result['means']  # (N, 3)
    if result['harmonics'] is not None:
        # SH 第 0 階（直流項）就是基礎顏色
        colors = result['harmonics'][..., :3, 0]  # (N, 3)
    else:
        # 無 SH 時，使用默認灰色
        colors = np.ones((len(means_np), 3)) * 128  # [128, 128, 128]
    
    # ✅ 第七步：色彩標準化到 [0, 255] uint8
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    
    result['colors'] = colors  # (N, 3) uint8 [0, 255]
    
    return result
```

**轉換後的結果**：

```python
gaussians_np = {
    'means': (N, 3) numpy float32,        # 世界座標粒子中心
    'scales': (N, 3) numpy float32,       # 縮放
    'rotations': (N, 4) numpy float32,    # 四元數
    'harmonics': numpy float32,           # SH 係數
    'opacities': (N,) numpy float32,      # 不透明度
    'colors': (N, 3) uint8 [0, 255]       # ⭐ RGB 顏色
}
```

### 第 6 層：Viser 點雲更新

**位置**: `src/pipeline/viser_renderer.py` → `update_point_cloud()`

**輸入**：
```python
viser_renderer.update_point_cloud(
    points=gaussians_np['means'],        # (N, 3) float32 世界座標
    colors=gaussians_np['colors'],       # (N, 3) uint8 [0, 255] RGB
    name="realtime_3dgs"                 # 場景名稱
)
```

**Viser 內部操作**：

```python
def update_point_cloud(self, points, colors, name="scene"):
    with self._lock:
        # 1. 色彩格式確認
        if colors.dtype == np.float32 or colors.dtype == np.float64:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
        
        # 2. 應用 UI 限制（最大點數）
        max_pts = int(self._gui_max_points.value * 1000)  # UI 滑塊
        if len(points) > max_pts:
            # 隨機採樣以滿足限制
            idx = np.random.choice(len(points), max_pts, replace=False)
            points = points[idx]
            colors = colors[idx]
        
        # 3. 建立 Viser 點雲（WebGL 呈現）
        self._server.scene.add_point_cloud(
            f"/{name}/points",                          # 場景路徑
            points=points.astype(np.float32),           # (N, 3) 位置
            colors=colors,                              # (N, 3) uint8 RGB
            point_size=self._gui_point_size.value,      # 動態大小滑塊
            point_shape="rounded"                        # 圓形點
        )
        
        # 4. 更新 UI 統計
        self._frame_count += 1
        self._gui_points.value = f"{len(points):,}"
```

### 第 7 層：Web 瀏覽器 3D 呈現

**輸出**：WebGL 3D 場景

```
http://0.0.0.0:8080
└── WebGL 3D Viewer
    ├── 點雲
    │   ├── 每個頂點位置：points[i] = means[i]
    │   ├── 每個頂點顏色：colors[i] = SH DC 項或默認灰色
    │   └── 點大小：0.005 (動態調整)
    ├── UI 控制面板
    │   ├── 信賴度閾值
    │   ├── 點大小滑塊
    │   └── 最大點數限制
    └── 相機視圖 (可交互)
```

---

## 🎯 粒子屬性對應表

| 粒子屬性 | 來源 | 計算方式 | 最終值 |
|---------|------|---------|--------|
| **位置 (X, Y, Z)** | `means` | 由深度反投影 + 世界座標變換 | (N, 3) float32 |
| **顏色 (R, G, B)** | `harmonics[..., :3, 0]` | SH 直流項，或默認 128 | (N, 3) uint8 0-255 |
| **大小** | `point_size` | Viser UI 滑塊 | 0.005 (動態) |
| **縮放 (Sx, Sy, Sz)** | `scales` | 考慮深度的 Gaussian 縮放 | (N, 3) float32 (未用到) |
| **旋轉** | `rotations` | 四元數 wxyz | (N, 4) float32 (未用到) |
| **不透明度** | `opacities` | 從概率密度計算 | (N,) float32 0-1 (未用到) |

**注意**：標記「未用到」的屬性在 Viser 的簡單點雲呈現 中暫未使用，但在高斯濺潑專業渲染器（如 gsplat）中會使用。

---

## 📊 數據流完整圖

```
DA3 模型 (GPU)
├─ output.depth: (1, 2, H', W')
├─ output.conf: (1, 2, H', W')
└─ output.gaussians: Gaussians {          ⭐ 核心
    means: (1, N, 3) torch [GPU]
    scales: (1, N, 3) torch [GPU]
    rotations: (1, N, 4) torch [GPU]
    harmonics: (1, N, 3, d_sh) torch [GPU]  ← 顏色來源
    opacities: (1, N,) torch [GPU]
   }
        ↓
DepthDecoder
└─ 保留 gaussians 原樣傳遞
        ↓
RenderThread: _process_da3_gaussians()
├─ .detach().cpu().numpy() 轉換
├─ 提取 means → (N, 3) numpy float32
├─ 提取 harmonics → (N, 3, d_sh) numpy
├─ 從 harmonics[..., :3, 0] 提取顏色
└─ 顏色標準化 → (N, 3) uint8 [0, 255]
        ↓
Viser 點雲更新
├─ points = means (N, 3)
├─ colors = harmonics DC (N, 3) uint8
└─ 應用 UI 限制 (max_points)
        ↓
WebGL 3D 場景
└─ 瀏覽器呈現粒子

🎯 最終用戶看到的粒子 =
    位置: means[i]
    顏色: harmonics[i, :3, 0]
    大小: UI 滑塊 0.005
```

---

## 🔍 色彩的詳細說明

### SH (Spherical Harmonics) 球諧係數

DA3 使用球諧函數來表示每個高斯粒子的色彩。這允許色彩隨著視角變化而變化（視圖依賴色彩）。

```
harmonics shape: (batch, N, 3, d_sh)
                        ↓ ↓
                      RGB, SH 階數
```

**SH 階數構成**：
- 0 階（直流項）: 3 個係數 (R, G, B) → **基礎顏色**
- 1-3 階: 另外 (3×4)² - 3 = 12 個係數 → 視圖依賴色彩調整

**Viser 中的簡化**：
```python
# Viser 只使用 0 階 SH（直流項）
colors = harmonics[..., :3, 0]  # shape: (N, 3)
# 這等於模型預測的基礎 RGB 顏色，不考慮視角變化
```

### 沒有 SH 時的備用處理

```python
if harmonics_available:
    colors = harmonics[..., :3, 0]
else:
    colors = np.ones((N, 3)) * 128  # 灰色 [128, 128, 128]
```

---

## 💡 性能因素

| 階段 | 最大粒子數 | 限制因素 |
|------|-----------|---------|
| DA3 預測 | 無上限 | GPU 內存 |
| 轉換 (NumPy) | 無上限 | CPU 內存 |
| Viser 呈現 | UI 限制 500k | `max_points` 滑塊 |
| 網頁傳遞 | ~100k 推薦 | 網絡頻寬 + WebGL |
| 性能 (60 FPS) | ~50k | 瀏覽器 GPU |

**當前配置**：
```yaml
gaussian:
  max_points: 500000  # 無作用（Viser 側有 UI 限制）

viser:
  point_size: 0.005   # 默認大小，可通過 UI 調整
```

---

## 📝 核心代碼引用

| 組件 | 檔案 | 函數/類 | 關鍵行 |
|------|------|--------|--------|
| Gaussians 定義 | specs.py | class Gaussians | L23 |
| GaussianAdapter | gs_adapter.py | GaussianAdapter.forward() | L58-164 |
| 轉換到世界座標 | gs_adapter.py | forward() | L68-82 (means) |
| SH 旋轉 | gs_adapter.py | forward() | L152 |
| 推理輸出 | inference_engine.py | _infer_pytorch() | L279 |
| 深度解碼 | depth_decoder.py | decode() | L28-77 |
| GPU→NumPy | realtime_pipeline.py | _process_da3_gaussians() | L57-141 |
| 顏色提取 | realtime_pipeline.py | _process_da3_gaussians() | L127-130 |
| Viser 更新 | viser_renderer.py | update_point_cloud() | L110-145 |
| 渲染入口 | realtime_pipeline.py | render_thread() | L260-285 |

---

---

## 🔀 粒子來源決策: 高斯頭 vs 深度投影

### 完整決策流程

```
配置文件 pipeline_config.yaml
├─ inference.use_gaussian_head: true   ← 🎯 當前配置
└─ inference.use_gaussian_head: false  ← 備用模式
        ↓
InferenceEngine 初始化
├─ use_gaussian_head = inf_cfg.get("use_gaussian_head")
└─ 根據此標誌決定推理時是否提取高斯頭
        ↓
DA3 模型推理
├─ 當 use_gaussian_head=True:
│  └─ output.gaussians = GaussianAdapter(...)  ✅ 高斯頭輸出
│
└─ 當 use_gaussian_head=False:
   └─ output.gaussians = None                ❌ 無高斯頭
        ↓
DepthDecoder.decode()
├─ inference_result['gaussians'] = output.gaussians
└─ (保留原樣，可能是 None)
        ↓
RenderThread 的關鍵決策點 (第 295 行)
```

### 🎯 路徑 A：直接使用高斯頭（當前配置）

```python
if use_gaussian_head and decoded.get("gaussians") is not None:
    # ✅ 當前使用的路徑
    
    gaussians_obj = decoded["gaussians"]  # 來自 DA3 模型
    gaussians_np = _process_da3_gaussians(gaussians_obj)
    
    # 粒子來源：直接從高斯頭
    points = gaussians_np["means"]        # (N, 3) DA3 預測的世界座標
    colors = gaussians_np["colors"]       # (N, 3) harmonics[..., :3, 0]
    
    viser_renderer.update_point_cloud(points, colors)
```

**粒子特性**：
| 屬性 | 來源 | 說明 |
|-----|------|------|
| **位置** | `means` | GaussianAdapter 計算的世界座標 |
| **顏色** | `harmonics[..., :3, 0]` | SH 直流項（模型預測的基礎顏色） |
| **數量** | 由深度圖解析度決定 | 通常 H'×W' 個點 |
| **計算時間** | GPU 推理 | 已包含在推理時間內 |

**流程圖**：
```
深度圖 + 原始圖片 (已轉換為 GPU 特徵)
    ↓
DA3 骨幹網 (DinoV2 feature extractor)
    ↓
DPT 頭 (深度估計)
    ↓
GSDPT 頭 (高斯估計 - 激活)
    ↓
GaussianAdapter (特徵→世界座標參數)
    ↓
Gaussians 對象 (means, scales, rotations, harmonics, opacities)
    ↓
轉換→Viser
```

### ❌ 路徑 B：深度投影（備用/禁用 gaussian_head）

```python
else:
    # 備用路徑：當 use_gaussian_head=False 或 gaussians=None
    
    depth_left = decoded["depth_left"]        # H'×W' 深度圖
    color_left = decoded["color_image_left"]  # H'×W'×3 原始 RGB
    K_l = pose_manager.get_left_intrinsic()   # 3×3 相機內參
    ext_l = pose_manager.get_left_extrinsic() # 4×4 相機外參
    
    gaussians = gaussian_projector.project(
        depth=depth_left,
        color_image=color_left,
        intrinsic=K_l,
        extrinsic=ext_l,
        confidence=conf_left,
        mask=conf_mask,
    )
    
    points = gaussians["means"]  # 反投影的 3D 座標
    colors = gaussians["colors"]  # 原始圖片的 RGB
    
    viser_renderer.update_point_cloud(points, colors)
```

**粒子特性**：
| 屬性 | 來源 | 說明 |
|-----|------|------|
| **位置** | 深度反投影 | `K_inv @ (u,v,1) * depth` + 外參變換 |
| **顏色** | 原始圖片 | `color_image_left[v, u]` |
| **數量** | 深度圖有效像素 | 受信心度遮罩限制 |
| **計算時間** | ~10-20ms | CPU 上的投影計算 |

**流程圖**：
```
深度圖 + 原始圖片
    ↓
GaussianProjector.project()
├─ 反投影深度到 3D → means
├─ 計算法線 + 旋轉 → rotations
├─ 深度梯度 → scales
└─ 圖片顏色 → colors
    ↓
轉換→Viser
```

### 🔍 兩者對比

| 維度 | 高斯頭（路徑 A） | 深度投影（路徑 B） |
|-----|-----------------|------------------|
| **粒子來源** | DA3 模型直接輸出 | 深度反投影計算 |
| **顏色來源** | SH 直流項 | 原始圖片 RGB |
| **計算位置** | GPU （推理內） | CPU （後處理） |
| **計算時間** | 0ms （包含在推理內） | 10-20ms |
| **精度** | 受模型訓練影響 | 取決於深度質量 |
| **視圖依賴** | ✅ 支持（通過 SH） | ❌ 固定色彩 |
| **高斯參數** | 完整 (means, scales, rotations, harmonics, opacities) | 部分 (means, scales, rotations, colors) |
| **需要相機參數** | ❌ 不需要（已在推理包含） | ✅ 需要內外參 |

### 📋 當前配置選擇

```yaml
# pipeline_config.yaml (當前)
inference:
  use_gaussian_head: true  ← 選擇 路徑 A （直接使用高斯頭）
  camera_params_mode: "auto"
```

**結果**：
```
✅ 使用 DA3 原生的高斯參數
✅ 粒子直接從 GaussianAdapter 輸出
❌ 不使用深度反投影
❌ 不需要精確相機標定
```

若要切換到路徑 B：
```yaml
inference:
  use_gaussian_head: false  ← 選擇 路徑 B （深度投影）
  camera_params_mode: "provided"  # 需要校正的相機參數
```

---

## ❓ 常見問題

### Q: 當前的 Viser 粒子是從高斯頭還是深度投影得到？
**A**: **直接從高斯頭**（因為 `use_gaussian_head: true`）。粒子位置來自 `means`（GaussianAdapter 輸出），顏色來自 `harmonics[..., :3, 0]`（SH 直流項）。深度圖和原始圖片在此模式下不用於粒子生成，只用於模型推理的輸入。

### Q: 如果關閉高斯頭會發生什麼？
**A**: 如果設 `use_gaussian_head: false` 或高斯頭輸出為 None，系統會自動切換到路徑 B，使用 GaussianProjector 將深度圖投影為點雲。此時需要校正的相機參數。

### Q: 為什麼粒子有時是灰色？
**A**: 當 DA3 的 `harmonics` 輸出為 None 或無效時，系統會使用默認灰色 `[128, 128, 128]`。這通常表示高斯頭未正確初始化。

### Q: 如何改變粒子大小？
**A**: 在 Viser 控制面板的「點大小」滑塊調整，範圍 0.001-0.05。

### Q: 粒子位置是世界座標嗎？
**A**: 是的。`means` 通過 GaussianAdapter 經過世界座標變換，確保所有粒子都在統一的世界座標系中。

### Q: 為什麼顏色不隨視角變化？
**A**: Viser 的簡單點雲渲染只使用 SH 的直流項（0 階），因此顏色固定。要看視圖依賴色彩，需要使用專業 3DGS 渲染引擎（如 gsplat）。

### Q: 粒子數量多少？
**A**: 在高斯頭模式下，粒子數 N 等於 output 的空間解析度（通常為 process_res = 504 的倍數），可能在 100k-500k 之間。Viser 中會通過 UI 滑塊限制最多顯示 500k 點。

