"""Cấu hình mặc định của hệ thống. Mọi ngưỡng đều có thể chỉnh trong trang Cài đặt
của dashboard (lưu vào data/settings.json)."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SNAP_DIR = DATA_DIR / "snapshots"
CLIP_DIR = DATA_DIR / "clips"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "events.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
CAMERAS_PATH = DATA_DIR / "cameras.json"

for _d in (DATA_DIR, SNAP_DIR, CLIP_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Lớp COCO được coi là phương tiện. COCO không có xe nâng / máy xúc:
# muốn nhận diện các lớp đó cần fine-tune (xem README).
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# Chiều cao thật trung bình (m) dùng cho tự hiệu chỉnh tỉ lệ khi chưa có homography
CLASS_HEIGHT_M = {"person": 1.70, "bicycle": 1.10, "car": 1.50, "motorcycle": 1.20,
                  "bus": 3.20, "truck": 3.00}


@dataclass
class FallSettings:
    descent_speed: float = 0.8        # tốc độ rơi của hông, đơn vị H_ref / giây
    height_drop: float = 0.35         # tỉ lệ chiều cao giảm trong cửa sổ descent_window
    descent_window: float = 0.7       # giây
    lying_angle: float = 60.0         # góc thân (độ) coi là nằm
    upright_angle: float = 35.0       # góc thân coi là đứng
    lying_aspect: float = 1.3         # AR = w/h của bbox coi là nằm (dự phòng)
    confirm_window: float = 2.0       # từ lúc rơi tới lúc nằm tối đa (giây)
    lying_hold: float = 1.0           # nằm liên tục bao lâu thì xác nhận FALL
    immobile_spread: float = 0.04     # độ dao động vị trí/kích thước bbox (theo H_ref) dưới ngưỡng = bất động
    torso_rate: float = 25.0          # thân đổ ≥ 25° trong 0,5 s cũng tính là pha rơi
    immobile_after: float = 10.0      # bất động sau ngã bao lâu thì nâng mức
    person_down_after: float = 5.0    # nằm bất động không có pha rơi -> PERSON_DOWN
    recover_hold: float = 1.5         # đứng lại bao lâu thì kết thúc sự kiện


@dataclass
class CollisionSettings:
    near_distance: float = 2.0        # m
    near_ttc: float = 2.0             # s
    near_hold: float = 0.3            # điều kiện phải kéo dài (s)
    min_vehicle_speed: float = 0.5    # m/s, xe đứng yên không tính near-miss theo khoảng cách
    contact_distance: float = 0.4     # m, người - xe
    confirm_window: float = 2.0       # s, chờ dấu hiệu bất thường của người sau tiếp xúc
    push_accel: float = 4.0           # m/s^2, người bị đẩy
    vv_contact_distance: float = 0.3  # m, xe - xe
    vv_min_closing: float = 1.0       # m/s trước va chạm
    vv_speed_drop: float = 0.5        # tỉ lệ giảm tốc sau va chạm
    vv_heading_change: float = 30.0   # độ
    cooldown: float = 10.0            # s, không báo lặp cùng một cặp


@dataclass
class PipelineSettings:
    process_fps: float = 10.0         # số khung hình phân tích mỗi giây
    imgsz: int = 640
    person_conf: float = 0.35
    vehicle_conf: float = 0.35
    pose_model: str = "yolo11n-pose.pt"
    det_model: str = "yolo11n.pt"
    focal_ratio: float = 0.9          # f ≈ 0.9 × chiều rộng ảnh (tự hiệu chỉnh)
    clip_pre: float = 3.0             # giây video trước sự kiện
    clip_post: float = 3.0            # giây sau sự kiện
    show_bev: bool = True             # bản đồ nhìn từ trên xuống ở góc khung hình
    realtime_files: bool = True       # phát file video với tốc độ thật


@dataclass
class Settings:
    fall: FallSettings = field(default_factory=FallSettings)
    collision: CollisionSettings = field(default_factory=CollisionSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, data: dict) -> None:
        for section in ("fall", "collision", "pipeline"):
            obj = getattr(self, section)
            for k, v in (data.get(section) or {}).items():
                if hasattr(obj, k):
                    cur = getattr(obj, k)
                    try:
                        setattr(obj, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
                    except (TypeError, ValueError):
                        pass


_lock = threading.Lock()
SETTINGS = Settings()


def load_settings() -> Settings:
    if SETTINGS_PATH.exists():
        try:
            SETTINGS.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except Exception:  # file hỏng -> dùng mặc định
            pass
    return SETTINGS


def save_settings() -> None:
    with _lock:
        SETTINGS_PATH.write_text(json.dumps(SETTINGS.to_dict(), indent=2, ensure_ascii=False),
                                 encoding="utf-8")
