# M3bot RL Training — Hướng Dẫn Sử Dụng

## Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu |
|---|---|
| **Isaac Sim** | 4.5+ (NVIDIA Omniverse) |
| **IsaacLab** | v2.3.2+ (thư mục `IsaacLab/`) |
| **rsl-rl-lib** | >= 3.0.1 |
| **Python** | 3.10 (đi kèm Isaac Sim) |
| **GPU** | NVIDIA RTX (CUDA, driver >= 535) |

---

## Cấu Trúc Dự Án

```
m3robot/
├── train.py                    ← script training chính
├── play.py                     ← script chạy & đánh giá policy
├── ARM/
│   └── M3bot.usd               ← model robot (4-DOF + gripper)
├── m3_tasks/
│   ├── __init__.py             ← đăng ký 8 Gymnasium envs
│   ├── m3_robot_cfg.py         ← config robot (actuator, camera)
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py   ← PPO hyperparameter configs
│   ├── m3_reach/               ← task: Reach
│   ├── m3_lift/                ← task: Lift
│   ├── m3_push/                ← task: Push
│   └── m3_pick_place/          ← task: Pick & Place
└── logs/
    └── rsl_rl/                 ← checkpoints & tensorboard logs
```

---

## Danh Sách Tasks

### State-based (obs vector phẳng — train nhanh)

| Task ID | Mô tả | Obs dim | Envs mặc định | Max iterations |
|---|---|---|---|---|
| `Isaac-M3-Reach-v0` | Đưa EE đến pose được chỉ định | 19 | 4096 | 1500 |
| `Isaac-M3-Lift-v0` | Nhặt cube và di chuyển đến goal 3D | 28 | 4096 | 3000 |
| `Isaac-M3-Push-v0` | Đẩy cube đến goal XY trên sàn | 22 | 4096 | 2000 |
| `Isaac-M3-PickPlace-v0` | Gắp và đặt cube vào vị trí đích | 27 | 4096 | 4000 |

### Vision-augmented (state + camera RGB 128×128 — cần nhiều VRAM hơn)

| Task ID | Envs mặc định |
|---|---|
| `Isaac-M3-Reach-Vision-v0` | 256 |
| `Isaac-M3-Lift-Vision-v0` | 256 |
| `Isaac-M3-Push-Vision-v0` | 256 |
| `Isaac-M3-PickPlace-Vision-v0` | 256 |

---

## Training

### Cú pháp cơ bản

```bash
./IsaacLab/isaaclab.sh -p train.py --task <TASK_ID> [options]
```

### Chế độ Headless (không cần màn hình — nhanh nhất)

```bash
# Reach — headless, dùng setting mặc định
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Reach-v0 --headless

# Lift — headless với ít env hơn (tiết kiệm VRAM)
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Lift-v0 --headless --num_envs 2048

# Push — headless, ghi video mỗi 2000 bước
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Push-v0 --headless --video --video_interval 2000

# Pick & Place — headless, tăng số iterations
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-PickPlace-v0 --headless --max_iterations 5000
```

### Chế độ Livestream (xem qua trình duyệt)

```bash
# Mở http://localhost:8211 sau khi chạy lệnh này
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Reach-v0 --livestream 2
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Lift-v0 --livestream 2
```

> **Lưu ý**: `--livestream 1` = OpenXR/native viewer, `--livestream 2` = WebRTC (trình duyệt tại localhost:8211)

### Resume từ checkpoint

```bash
# Tự động tìm checkpoint mới nhất trong logs/
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Lift-v0 --headless --resume

# Chỉ định checkpoint cụ thể
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Lift-v0 --headless \
    --checkpoint logs/rsl_rl/m3_lift/2025-01-01_12-00-00/model_1000.pt
```

### Tất cả options của train.py

| Flag | Mặc định | Mô tả |
|---|---|---|
| `--task` | (bắt buộc) | Tên task Gymnasium |
| `--headless` | false | Không mở GUI |
| `--livestream 2` | off | WebRTC stream tại :8211 |
| `--num_envs N` | theo task | Số env song song |
| `--max_iterations N` | theo task | Số iterations PPO |
| `--seed N` | 42 | Random seed |
| `--video` | false | Ghi video trong lúc train |
| `--video_length N` | 200 | Độ dài video (steps) |
| `--video_interval N` | 2000 | Khoảng cách giữa các video |
| `--resume` | false | Tiếp tục từ checkpoint mới nhất |
| `--checkpoint PATH` | auto | Đường dẫn checkpoint cụ thể |
| `--logger` | tensorboard | Backend log (tensorboard/wandb/neptune) |
| `--experiment_name` | theo task | Tên folder trong logs/ |
| `--run_name` | - | Suffix tên run |

### Vị trí logs

```
logs/rsl_rl/
├── m3_reach/
│   └── 2025-01-01_12-00-00/
│       ├── params/env.yaml       ← config environment
│       ├── params/agent.yaml     ← config PPO
│       ├── model_500.pt          ← checkpoint mỗi 50 iters
│       ├── model_1000.pt
│       └── videos/train/         ← video nếu dùng --video
├── m3_lift/
├── m3_push/
└── m3_pick_place/
```

### Xem TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/m3_reach
# Mở http://localhost:6006
```

---

## Play / Đánh Giá

### Cú pháp cơ bản

```bash
./IsaacLab/isaaclab.sh -p play.py --task <TASK_ID> [options]
```

> **Yêu cầu**: Phải có checkpoint đã train trong `logs/rsl_rl/<task>/`. Nếu chưa train, chạy `train.py` trước.

### Visualize với GUI (mặc định)

```bash
# Tự động load checkpoint mới nhất
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Reach-v0
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Lift-v0
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Push-v0
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-PickPlace-v0

# Ít envs hơn để quan sát rõ hơn
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Lift-v0 --num_envs 4
```

### Ghi video và thoát

```bash
# Ghi 300 bước và thoát
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Lift-v0 --video --video_length 300

# Headless + video (nhanh nhất)
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Push-v0 --headless --video

# Video lưu tại:
# logs/rsl_rl/m3_lift/<run_timestamp>/videos/play/
```

### Livestream qua trình duyệt

```bash
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Reach-v0 --livestream 2
# Mở http://localhost:8211
```

### Real-time playback

```bash
# Làm chậm simulation về tốc độ thực
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Reach-v0 --real_time
```

### Chỉ định checkpoint cụ thể

```bash
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Lift-v0 \
    --checkpoint logs/rsl_rl/m3_lift/2025-01-01_12-00-00/model_3000.pt
```

### Tất cả options của play.py

| Flag | Mặc định | Mô tả |
|---|---|---|
| `--task` | (bắt buộc) | Tên task Gymnasium |
| `--headless` | false | Không mở GUI |
| `--livestream 2` | off | WebRTC stream tại :8211 |
| `--num_envs N` | 4 | Số env khi eval |
| `--seed N` | 0 | Random seed |
| `--checkpoint PATH` | auto | Đường dẫn checkpoint (tự tìm nếu bỏ trống) |
| `--video` | false | Ghi video và thoát sau `video_length` bước |
| `--video_length N` | 300 | Số bước ghi video |
| `--real_time` | false | Làm chậm về tốc độ thực |
| `--export_policy` | true | Xuất JIT + ONNX khi load checkpoint |

### Policy export

Khi `play.py` chạy, policy tự động được xuất vào:

```
logs/rsl_rl/<task>/<run>/exported/
├── policy.pt      ← TorchScript JIT (dùng deploy trên robot)
└── policy.onnx    ← ONNX (dùng với TensorRT, ONNX Runtime)
```

---

## Vision Tasks

Vision tasks thêm camera RGB 128×128 vào observation. Cần bật camera khi chạy:

```bash
# Train vision task — BẮT BUỘC thêm --enable_cameras
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Reach-Vision-v0 --headless --enable_cameras

# Play vision task
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Lift-Vision-v0 --enable_cameras --num_envs 4
```

> **VRAM**: Vision tasks cần ~2× VRAM. Giảm `num_envs` nếu hết bộ nhớ.

---

## PPO Hyperparameters (Tham Khảo)

| Task | actor/critic dims | iterations | envs | num_steps |
|---|---|---|---|---|
| Reach | [128, 64, 64] | 1500 | 4096 | 24 |
| Lift | [256, 128, 64] | 3000 | 4096 | 24 |
| Push | [128, 64, 64] | 2000 | 4096 | 24 |
| PickPlace | [256, 128, 64] | 4000 | 4096 | 24 |

Tất cả dùng:
- Optimizer: Adam, lr=1e-3, adaptive schedule
- Clip param: 0.2, entropy coef: 0.005
- Discount: γ=0.99, λ=0.95 (GAE)
- Observation normalization: bật (actor + critic)

Chỉnh trong `m3_tasks/agents/rsl_rl_ppo_cfg.py`.

---

## Thiết Lập Môi Trường (Lần Đầu)

```bash
# 1. Cài IsaacLab dependencies
cd IsaacLab
./isaaclab.sh --install

# 2. Cài rsl-rl
./isaaclab.sh -p -m pip install rsl-rl-lib

# 3. Kiểm tra cài đặt
./isaaclab.sh -p -c "import rsl_rl; print(rsl_rl.__version__)"

# 4. Chạy test syntax (không cần GPU)
python3 -c "
import ast, os
for root, _, files in os.walk('m3_tasks'):
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            ast.parse(open(p).read())
            print('OK', p)
print('PASSED')
"
```

---

## Thứ Tự Khuyên Dùng Khi Bắt Đầu

1. **Reach** — task đơn giản nhất, train nhanh (~30 phút với 4096 envs)
2. **Push** — không cần gripper, train vừa (~1 tiếng)
3. **Lift** — cần học gripper, phức tạp hơn (~2 tiếng)
4. **PickPlace** — task khó nhất, kết hợp lift + push (~3-4 tiếng)

```bash
# Chạy nhanh để kiểm tra env hoạt động (1 iteration, 64 envs)
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Reach-v0 \
    --headless --num_envs 64 --max_iterations 1
```

---

## Xử Lý Lỗi Thường Gặp

### `ModuleNotFoundError: No module named 'isaaclab'`
```bash
# Phải dùng isaaclab.sh, không dùng python trực tiếp
./IsaacLab/isaaclab.sh -p train.py ...
```

### `rsl-rl-lib X.Y.Z is installed but >=3.0.1 is required`
```bash
./IsaacLab/isaaclab.sh -p -m pip install rsl-rl-lib --upgrade
```

### CUDA out of memory
```bash
# Giảm số envs
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Lift-v0 --headless --num_envs 512
```

### `get_checkpoint_path: no checkpoint found`
```bash
# Phải train trước khi play
./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Reach-v0 --headless
# Sau đó mới chạy play
./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Reach-v0
```

### Livestream không kết nối được
```bash
# Đảm bảo port 8211 không bị block (firewall)
# Mở http://localhost:8211 (không phải https)
# Nếu chạy trên server remote: dùng SSH tunnel
ssh -L 8211:localhost:8211 user@server
```

---

## Test Kết Quả

| Kiểm tra | Kết quả |
|---|---|
| Syntax — 30 files (`m3_tasks/` + `train.py` + `play.py`) | ✅ PASSED |
| Custom MDP functions exported đúng trong tất cả 4 local mdp modules | ✅ PASSED |
| Tất cả 4 env_cfg dùng local mdp import (không dùng `isaaclab.envs.mdp`) | ✅ PASSED |
| Tất cả function definitions tồn tại trong rewards/observations/terminations | ✅ PASSED |
| PPO configs có đủ 4 classes + experiment names | ✅ PASSED |
| Task registry trong train.py và play.py cover đủ 8 tasks | ✅ PASSED |
| Asset file `ARM/M3bot.usd` tồn tại | ✅ PASSED |
