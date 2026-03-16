# DA3 Gaussian Head 管道流程分析

當 `use_gaussian_head: true` 時，從輸入到 Viser 輸出的完整數據流。

---

## 📊 完整流程圖

```
[視頻流] → [Capture Thread] → [Frame Buffer] → [Inference Thread] → 
[Inference Engine] → [Depth Decoder] → [Render Thread] → [Viser Renderer]
   ↓            ↓               ↓              ↓              ↓
 [立體校正]   [推送幀]      [時序整合]     [DA3推理]      [高斯轉換]
            可選                      Gaussians
```

---

## 🔄 各階段詳解

### 1️⃣ 輸入採集 (Capture Thread)
**位置**: `src/realtime_pipeline.py` → `capture_thread()`

#### 輸入：
- **視頻流 URL**: `http://localhost:5000/video_combined`
- **幀格式**: 側邊並排 (side-by-side) 的 BGR uint8 圖像
  - 左半部分 (MX Brio) + 右半部分 (RealSense)
  - 分辨率: `width × height` (預設 640×480)

#### 處理：
```python
# 1. 從 Flask 流讀取幀
cap = cv2.VideoCapture(stream_url)
ret, frame = cap.read()  # shape: (480, 1280, 3) uint8 BGR

# 2. 分割左右圖像
h, w = frame.shape[:2]
mid = w // 2
img_left = frame[:, :mid]      # (480, 640, 3)
img_right = frame[:, mid:]     # (480, 640, 3)

# 3. 可選的立體校正（使用標定參數）
if enable_rectification and rectifier is not None:
    img_left, img_right = rectifier.rectify(img_left, img_right)
```

#### 輸出：
- 校正後的左右 RGB uint8 圖像推送到循環緩衝區

| 項目 | 值 | 形狀 |
|-----|-----|------|
| 左幀 | uint8 BGR | (480, 640, 3) |
| 右幀 | uint8 BGR | (480, 640, 3) |
| 來源 | Flask stream | - |
| 標定 | calibration_params.yml | - |

---

### 2️⃣ 時序緩衝與整合 (Frame Buffer)
**位置**: `src/pipeline/frame_buffer.py` → `CircularFrameBuffer`

#### 輸入：
- **時序幀數**: `temporal_frames` (設定檔中的 buffer.temporal_frames)
- **緩衝幀** (來自 Capture Thread)

#### 批次組成 (動態，根據 temporal_frames)：

**例子1：temporal_frames=1 (當前配置)**
```
Batch = [L_t, R_t]
         ├─ 左相機 1 幀 ─┤  ├─ 右相機 1 幀 ─┤
         
batch_size = temporal_frames × 2 = 2 幀
capacity = temporal_frames = 1
```

**例子2：temporal_frames=3 (預設)**
```
Batch = [L_{t-2}, L_{t-1}, L_t, R_{t-2}, R_{t-1}, R_t]
         ├─────── 左相機 3 幀 ───────┤  ├─────── 右相機 3 幀 ───────┤
         
batch_size = temporal_frames × 2 = 6 幀
capacity = temporal_frames = 3
```

#### 幀索引計算（動態）：
```
IDX_LEFT_T = temporal_frames - 1      # 左相機當前幀索引
IDX_RIGHT_T = temporal_frames * 2 - 1 # 右相機當前幀索引

當 temporal_frames=1:
  IDX_LEFT_T = 0,  IDX_RIGHT_T = 1

當 temporal_frames=3:
  IDX_LEFT_T = 2,  IDX_RIGHT_T = 5
```

#### 輸出：
- `temporal_frames × 2` 張 uint8 BGR 幀的列表

| 項目 | 值 (temporal_frames=1) | 值 (temporal_frames=3) |
|-----|------|------|
| 總幀數 | 2 | 6 |
| 每幀大小 | (480, 640, 3) uint8 | (480, 640, 3) uint8 |
| 類型 | List[np.ndarray] | List[np.ndarray] |
| FrameBuffer.capacity | 1 | 3 |

---

### 3️⃣ 推理引擎 (Inference Engine)
**位置**: `src/pipeline/inference_engine.py` → `InferenceEngine.infer()`

#### 輸入：
```python
frames: List[np.ndarray]           # temporal_frames*2 幀, (480, 640, 3) uint8 BGR
extrinsics: Optional[torch.Tensor] # (temporal_frames*2, 4, 4) or None
intrinsics: Optional[torch.Tensor] # (temporal_frames*2, 3, 3) or None
```

**根據 camera_params_mode 決定**：
```python
if camera_params_mode == "provided":
    # 使用校定到的相機內外參
    extrinsics = pose_manager.get_batch_extrinsics()  # (temporal_frames*2, 4, 4)
    intrinsics = pose_manager.get_batch_intrinsics()  # (temporal_frames*2, 3, 3)
    print("[Inference] 使用校正後的相機內外參 (provided mode)")
else:
    # 當前設定："auto" — 不傳入內外參
    extrinsics = None
    intrinsics = None
    print("[Inference] 不傳入相機內外參，由模型自行估計 (auto mode)")
```

**目前配置**：
- `camera_params_mode: "auto"` ✅ 
- → extrinsics = **None**
- → intrinsics = **None**

#### 預處理：
```python
# 1. BGR → RGB，uint8 → float [0,1]
frame_rgb = frame[..., ::-1] / 255.0  # (480, 640, 3) float32

# 2. HWC → CHW
t = frame_rgb.permute(2, 0, 1)        # (3, 480, 640)

# 3. ImageNet 歸一化
t_norm = (t - MEAN) / STD              # (3, 480, 640) float32

# 4. 堆疊成 Batch，調整到 process_res
batch = torch.stack(tensors, dim=0)    # (1, temporal_frames*2, 3, H', W')
                                       # H', W' 對齐到 14 的倍數
                                       # process_res = 504
```

**當前配置 (temporal_frames=1)**：
```
batch shape: (1, 2, 3, H', W')   ← 2 幀 (L_t, R_t)
```

#### 推理執行：
```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = self._model.model(
        batch,                  # (1, temporal_frames*2, 3, H', W')
        extrinsics=None,        # camera_params_mode=="auto" ✅
        intrinsics=None         # camera_params_mode=="auto" ✅
    )
```

#### 輸出提取：
```python
# DA3 模型輸出
output.depth        # (1, temporal_frames*2, H', W')
output.depth_conf   # (1, temporal_frames*2, H', W')
output.gaussians    # ⭐ Gaussians 對象 (if use_gaussian_head=True)
```

**當前配置**：
```
depth shape: (1, 2, H', W')  ← 2 個深度圖 (L_t, R_t)
```

#### 推理結果字典：
| 鍵 | 形狀 | 類型 | 說明 |
|---|------|------|------|
| `depth` | (temporal_frames*2, H', W') | float32 numpy | 深度圖 (當前配置: 2 幀) |
| `conf` | (temporal_frames*2, H', W') | float32 numpy | 信心度圖 (當前配置: 2 幀) |
| **`gaussians`** | - | Gaussians object | ⭐ **高斯頭輸出** |
| `time_ms` | scalar | float | 推理時間 (毫秒) |

---

### 4️⃣ 深度解碼器 (Depth Decoder)
**位置**: `src/pipeline/depth_decoder.py` → `DepthDecoder.decode()`

#### 輸入：
```python
inference_result = {
    'depth': (temporal_frames*2, H', W') float32,
    'conf': (temporal_frames*2, H', W') float32,
    'gaussians': Gaussians object,  # ⭐ 核心
    'time_ms': float
}
```

#### 提取當前時刻幀（動態計算）：
```python
IDX_LEFT_T = temporal_frames - 1       # 左相機當前幀索引
IDX_RIGHT_T = temporal_frames * 2 - 1  # 右相機當前幀索引

depth_left = depth_all[IDX_LEFT_T]     # (H', W')
depth_right = depth_all[IDX_RIGHT_T]   # (H', W')
conf_left = conf_all[IDX_LEFT_T]       # (H', W')
conf_right = conf_all[IDX_RIGHT_T]     # (H', W')
gaussians = gaussians                  # ⭐ Gaussians物件保留不變
```

**當前配置 (temporal_frames=1)**：
```
IDX_LEFT_T = 0      → depth_left = depth[0]
IDX_RIGHT_T = 1     → depth_right = depth[1]
```

#### 信心度遮罩：
```python
confidence_threshold = 0.8  # 來自設定檔
mask_left = conf_left >= 0.8   # (H', W') bool
mask_right = conf_right >= 0.8 # (H', W') bool
```

#### 解碼輸出：
```python
{
    'depth_left': (H', W') float32,
    'depth_right': (H', W') float32,
    'conf_left': (H', W') float32,
    'conf_right': (H', W') float32,
    'mask_left': (H', W') bool,
    'mask_right': (H', W') bool,
    'depth_all': (6, H', W') float32,
    'gaussians': Gaussians object,  # ⭐ 保留給渲染線程
    'color_image_left': (H, W, 3) uint8 BGR,   # 儲存原始幀用於著色
    'color_image_right': (H, W, 3) uint8 BGR,
    'time_ms': float
}
```

| 項目 | 值 | 用途 |
|-----|-----|------|
| 解碼深度 | (H', W') | 備用（use_gaussian_head=True 時不用） |
| **Gaussians** | object | ⭐ **直接傳給渲染線程** |
| 原始幀 | (H, W, 3) uint8 | 如需備用時著色 |

---

### 5️⃣ 渲染線程 (Render Thread) - **高斯頭路徑**
**位置**: `src/realtime_pipeline.py` → `render_thread()` 

#### 🎯 關鍵決策點
```python
if use_gaussian_head and decoded.get("gaussians") is not None:
    # ✅ 走這條路：使用 DA3 原生高斯
else:
    # ❌ 備用路：使用深度投影（需要 GaussianProjector）
```

#### 5.1 高斯處理 (use_gaussian_head=True)

**輸入**:
```python
gaussians_obj = decoded["gaussians"]  # ⭐ Gaussians 對象在 GPU 上
```

**Gaussians 對象結構**:
```python
gaussians_obj.means         # (N, 3) torch.Tensor, 高斯中心位置 [GPU]
gaussians_obj.scales        # (N, 3) torch.Tensor, 縮放因子
gaussians_obj.rotations     # (N, 4) torch.Tensor, 四元數 (w, x, y, z)
gaussians_obj.harmonics     # (B, N, 3, d_sh), SH 球諧係數 (RGB × SH階數)
gaussians_obj.opacities     # (N,) torch.Tensor, 不透明度 [0, 1]
```

**轉換函數**: `_process_da3_gaussians()`

```python
def _process_da3_gaussians(gaussians_obj):
    """GPU → NumPy 轉換，提取可渲染的數據"""
    
    result = {}
    
    # 1. 提取位置 (必需)
    result['means'] = gaussians_obj.means.detach().cpu().numpy()
    # 形狀: (N, 3) float32
    
    # 2. 提取縮放
    result['scales'] = gaussians_obj.scales.detach().cpu().numpy()
    # 形狀: (N, 3) float32
    
    # 3. 提取旋轉
    result['rotations'] = gaussians_obj.rotations.detach().cpu().numpy()
    # 形狀: (N, 4) float32 (四元數)
    
    # 4. 提取 SH 諧波係數
    result['harmonics'] = gaussians_obj.harmonics.detach().cpu().numpy()
    # 形狀: (B, N, 3, d_sh) 或類似
    
    # 5. 提取不透明度
    result['opacities'] = gaussians_obj.opacities.detach().cpu().numpy()
    # 形狀: (N,) float32
    
    # 6. 從 SH 或默認生成顏色
    if harmonics available:
        # 使用 SH 的直流項 (DC term) 計算顏色
        colors = harmonics[..., :3, 0]  # DC = 0階 SH
    else:
        # 無 SH 時用灰色
        colors = np.ones((N, 3)) * 128
    
    # 確保色彩範圍 [0, 255]
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    
    result['colors'] = colors  # (N, 3) uint8
    
    return result
```

**轉換過程表**:

| 階段 | 輸入 | 輸出 | 備註 |
|------|------|------|------|
| GPU張量 | `gaussians.means` torch (N,3) [GPU] | - | float32 |
| ↓ detach | `means.detach()` | - | 斷開計算圖 |
| ↓ CPU | `.cpu()` | - | 移到 CPU |
| ↓ NumPy | `.numpy()` | `means_np` (N,3) float32 | CPU NumPy |
| 顏色生成 | `harmonics` 或默認 | `colors` (N,3) uint8 [0,255] | 可渲染 |

#### 5.2 渲染輸入彙總

在 `update_point_cloud()` 呼叫時：

```python
viser_renderer.update_point_cloud(
    points=gaussians_np['means'],          # (N, 3) float32
    colors=gaussians_np['colors'],         # (N, 3) uint8 [0,255]
    name="realtime_3dgs"
)
```

| 參數 | 值 | 形狀 | 類型 |
|-----|-----|------|------|
| **points** | Gaussian 中心位置 | (N, 3) | float32 |
| **colors** | RGB 顏色 (SH 或默認) | (N, 3) | uint8 |
| **name** | 場景物體名稱 | - | "realtime_3dgs" |

---

### 6️⃣ Viser 渲染器 (Viser Renderer)
**位置**: `src/pipeline/viser_renderer.py` → `ViserRenderer.update_point_cloud()`

#### 輸入：
```python
points   # (N, 3) float32 世界空間座標
colors   # (N, 3) uint8 RGB [0, 255]
name     # str 物體名稱
```

#### 渲染管道：
```python
def update_point_cloud(self, points, colors, name="scene"):
    
    # 1. 色彩格式標準化
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    
    # 2. 應用動態最大點數限制
    max_pts = int(self._gui_max_points.value * 1000)  # UI 滑塊控制
    if len(points) > max_pts:
        # 隨機採樣減少點數
        idx = np.random.choice(len(points), max_pts, replace=False)
        points = points[idx]
        colors = colors[idx]
    
    # 3. 建立 Viser 點雲
    self._server.scene.add_point_cloud(
        f"/{name}/points",
        points=points.astype(np.float32),           # (N, 3) float32
        colors=colors,                               # (N, 3) uint8
        point_size=self._gui_point_size.value,      # 動態點大小
        point_shape="rounded"                        # 圓形點
    )
    
    # 4. 更新 UI 統計
    self._frame_count += 1
    self._fps = fps_calculated
    self._gui_fps.value = f"{fps:.1f}"
    self._gui_points.value = f"{len(points):,}"
```

#### Viser UI 控制（動態）：
| 控制項 | 範圍 | 預設 | 效果 |
|-------|------|------|------|
| 信賴度閾值 | 0.0 ~ 1.0 | 0.8 | (備用深度投影時) |
| 點大小 | 0.001 ~ 0.05 | 0.005 | 調整視覺大小 |
| 最大點數 | 10k ~ 1000k | 500k | 性能 vs 細節 |

#### Web 瀏覽器輸出：
```
http://0.0.0.0:8080
├── 3D 場景
│   └── /realtime_3dgs/points
│       ├── 點雲 (N, 3) 位置
│       ├── RGB 顏色
│       └── 點渲染引擎
├── 控制面板
│   ├── 信賴度閾值滑塊
│   ├── 點大小滑塊
│   └── 最大點數滑塊
└── 狀態資訊
    ├── FPS
    └── 當前點數
```

---

## 📈 數據流完整表

```
┌─────────────────────────────────────────────────────────────────┐
│                     use_gaussian_head: true                      │
│              camera_params_mode: "auto" (當前配置)              │
│              temporal_frames: 1 (當前配置)                      │
└─────────────────────────────────────────────────────────────────┘

CAPTURE THREAD
├─ 輸入: Flask stream (side-by-side BGR)
│  └─ 形狀: (480, 1280, 3) uint8
│
├─ 立體校正 (可選)
│  └─ calibration_params.yml
│
└─ 輸出: [img_left, img_right]
   └─ 各 (480, 640, 3) uint8 → 推動到 FrameBuffer

FRAME BUFFER (capacity = temporal_frames = 1)
├─ 輸入: 逐幀的左右圖像
│
├─ 時序整合 (temporal_frames=1)
│  └─ Batch = [L_t, R_t]  ← 只有 2 幀 (不是 6 幀!)
│
└─ 輸出: List of 2 frames
   └─ 各 (480, 640, 3) uint8 → InferenceEngine

INFERENCE ENGINE (_infer_pytorch)
├─ 輸入:
│  ├─ frames: List[2 × (480, 640, 3) uint8 BGR]
│  ├─ extrinsics: None  ✅ (camera_params_mode="auto")
│  └─ intrinsics: None  ✅ (camera_params_mode="auto")
│
├─ 預處理:
│  ├─ BGR → RGB
│  ├─ uint8 → float [0, 1]
│  ├─ HWC → CHW
│  ├─ ImageNet 歸一化
│  └─ 調整到 process_res=504
│
├─ DA3 模型推理:
│  └─ output = model(batch)  ← 模型自行估計相機參數
│
└─ 輸出: {
     'depth': (2, H', W') float32,  ← 2 幀深度
     'conf': (2, H', W') float32,   ← 2 幀信心度
     'gaussians': Gaussians {            ⭐ CORE
         means: (N, 3) torch [GPU],
         scales: (N, 3) torch [GPU],
         rotations: (N, 4) torch [GPU],
         harmonics: (B, N, 3, d_sh) torch [GPU],
         opacities: (N,) torch [GPU]
     },
     'time_ms': float
   }

DEPTH DECODER
├─ 輸入: 推理結果
│  └─ depth (2, H', W'), conf (2, H', W'), gaussians
│
├─ 提取當前時刻 (IDX_LEFT_T=0, IDX_RIGHT_T=1):
│  ├─ depth_left = depth[0]
│  ├─ depth_right = depth[1]
│  └─ gaussians 保留原樣
│
├─ 計算遮罩:
│  └─ conf >= confidence_threshold
│
└─ 輸出: {
     'depth_left': (H', W') float32,
     'depth_right': (H', W') float32,
     'conf_left': (H', W') float32,
     'conf_right': (H', W') float32,
     'mask_left': (H', W') bool,
     'mask_right': (H', W') bool,
     'color_image_left': (H, W, 3) uint8,
     'color_image_right': (H, W, 3) uint8,
     'gaussians': Gaussians {        ⭐ PRESERVED
         means, scales, rotations, harmonics, opacities
     },
     'time_ms': float
   }

RENDER THREAD
├─ 輸入: 解碼結果
│
├─ 高斯轉換 [use_gaussian_head=True]:
│  └─ _process_da3_gaussians(gaussians_obj)
│     ├─ GPU → CPU (detach, cpu, numpy)
│     ├─ means: (N, 3) torch → numpy float32
│     ├─ scales: (N, 3) torch → numpy float32
│     ├─ rotations: (N, 4) torch → numpy float32
│     ├─ harmonics: torch → numpy (SH 係數)
│     ├─ opacities: torch → numpy float32
│     └─ colors: 由 harmonics DC 項或默認灰色
│        └─ (N, 3) uint8 [0, 255]
│
├─ 性能優化:
│  ├─ 應用最大點數限制 (UI 控制)
│  └─ 隨機採樣如需要降採樣
│
└─ 輸出: viser_renderer.update_point_cloud(
     points=(N, 3) float32,
     colors=(N, 3) uint8,
     name="realtime_3dgs"
   )

VISER RENDERER (Web 3D)
├─ 輸入:
│  ├─ points: (N, 3) float32 世界座標
│  ├─ colors: (N, 3) uint8 RGB
│  └─ point_size: 0.005 (動態)
│
├─ 場景建構:
│  └─ server.scene.add_point_cloud(
│     points, colors, point_size
│   )
│
├─ 互動 UI:
│  ├─ 信賴度閾值 (備用時)
│  ├─ 點大小調整
│  ├─ 最大點數限制
│  └─ FPS 和點數監測
│
└─ 輸出: HTTP 瀏覽器
   └─ http://0.0.0.0:8080 → WebGL 3D 呈現
```

---

## 🔑 核心數據項目彙表

### 高斯頭模式下的關鍵數據：

| 階段 | 數據名 | 形狀 | 類型 | GPU? | 說明 |
|-----|--------|------|------|------|------|
| **推理** | `output.gaussians` | - | Gaussians object | ✅ | DA3 原生高斯輸出 |
| **推理** | `output.gaussians.means` | (N, 3) | torch.Tensor | ✅ | 高斯中心位置 |
| **推理** | `output.gaussians.harmonics` | (B, N, 3, d_sh) | torch.Tensor | ✅ | SH 球諧顏色系數 |
| **推理** | `output.gaussians.opacities` | (N,) | torch.Tensor | ✅ | 不透明度 |
| **轉換** | `means_np` | (N, 3) | numpy float32 | ❌ | CPU NumPy 位置 |
| **轉換** | `colors_np` | (N, 3) | numpy uint8 | ❌ | uint8 RGB [0,255] 顏色 |
| **渲染** | `points` | (N, 3) | numpy float32 | ❌ | Viser 輸入點 |
| **渲染** | `colors` | (N, 3) | numpy uint8 | ❌ | Viser 輸入色彩 |

---

## ⚙️ 配置影響

```yaml
# pipeline_config.yaml (當前設定)

buffer:
  temporal_frames: 1  # 💡 決定批次大小: batch_size = 1×2 = 2 幀
                       # 不是 6 幀！(6 幀是 temporal_frames=3 時)

inference:
  camera_params_mode: "auto"  # 💡 模型自行估計相機參數
                              #     ✅ 不傳 extrinsics/intrinsics
                              #     轉換為 None
  
  use_gaussian_head: true     # 啟用 DA3 高斯頭

gaussian:
  scale_multiplier: 2.0       # (備用深度投影時，本模式忽略)
  max_points: 500000          # (無影響：Viser 側有獨立限制)
```

### temporal_frames 的影響

| 配置值 | 批次幀數 | 緩衝容量 | IDX_LEFT_T | IDX_RIGHT_T | 備註 |
|--------|---------|---------|-----------|-----------|------|
| 1      | 2       | 1       | 0         | 1         | **當前** ✅ |
| 2      | 4       | 2       | 1         | 3         | 當前幀+前一幀 |
| 3      | 6       | 3       | 2         | 5         | 預設(原文檔錯誤) |

### camera_params_mode 的影響

| 模式 | extrinsics | intrinsics | 說明 |
|-----|-----------|-----------|------|
| **"auto"** ✅ | None | None | **當前** — 模型自行估計相機參數 |
| "provided" | (2,4,4) tensor | (2,3,3) tensor | 使用標定的內外參 |

---

## 🎯 性能指標

典型情況（使用高斯頭）：

| 指標 | 值 | 單位 |
|-----|-----|------|
| 推理時間 | 100～200 | ms |
| 渲染時間 | 5～15 | ms |
| 輸出點數 (典型) | 50k～200k | points |
| 最大支持點數 | 500k～1M | points (UI受限) |
| Web FPS | 20～50 | fps (取決於點數和網絡) |

---

## � 文檔修正說明

### 修正項目 1：批次幀數是動態的

**錯誤版本**：  
文檔最初假設 `temporal_frames=3` (預設)，導出批次大小為 **6 幀**。

**正確版本**：  
批次大小由 `temporal_frames` 動態決定：
- **batch_size = temporal_frames × 2**
- 當前配置 (temporal_frames=1) → **2 幀** [L_t, R_t]
- 預設配置 (temporal_frames=3) → **6 幀** [L_{t-2}, ..., L_t, R_{t-2}, ..., R_t]

### 修正項目 2：camera_params_mode="auto" 不傳內外參

**錯誤版本**：  
文檔中的推理引擎部分沒有明確區分兩種模式。

**正確版本**：  
```python
if camera_params_mode == "provided":
    extrinsics = pose_manager.get_batch_extrinsics()
    intrinsics = pose_manager.get_batch_intrinsics()
else:  # "auto" (當前設定)
    extrinsics = None      # ✅ 不傳入
    intrinsics = None      # ✅ 不傳入
    # 模型自行估計相機參數
```

當前配置使用 **"auto" 模式** → extrinsics 和 intrinsics 確實是 **None**。

---

### use_gaussian_head = True (本文檔)
✅ 優點：
- 直接使用 DA3 預測的高斯參數
- 無需深度反投影計算
- 保留模型的旋轉和尺度信息
- 更符合模型訓練目標

❌ 缺點：
- 依賴模型輸出的 Gaussians 物件可用性
- 需要模型支持 gaussian_head

### use_gaussian_head = False (備用)
✅ 優點：
- 通用備用方案
- 支持任何深度估計模型

❌ 缺點：
- 需要計算深度梯度 → 法線 → 旋轉
- 深度反投影計算開銷
- 可能損失信息

---

## 📝 關鍵函數引用

| 函數 | 檔案 | 行號 | 用途 |
|-----|------|-------|--------|
| `_process_da3_gaussians()` | realtime_pipeline.py | ~57 | GPU→NumPy |
| `InferenceEngine.infer()` | inference_engine.py | ~170 | 執行推理 |
| `InferenceEngine._infer_pytorch()` | inference_engine.py | ~250 | PyTorch 後端 |
| `DepthDecoder.decode()` | depth_decoder.py | ~28 | 提取當前幀 |
| `render_thread()` | realtime_pipeline.py | ~209 | 渲染流程 |
| `ViserRenderer.update_point_cloud()` | viser_renderer.py | ~110 | Web 3D 更新 |

