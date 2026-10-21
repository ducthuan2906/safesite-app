"""Đọc video/camera → YOLO (pose cho người, detect cho xe) + ByteTrack → Engine → vẽ,
lưu sự kiện. Mỗi camera chạy trong một luồng riêng."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from .annotate import draw
from .config import DATA_DIR, SETTINGS, VEHICLE_CLASSES, Settings
from .engine import Engine
from .events import ClipRecorder, EventHub
from .tracks import Detection

TRACKER_YAML = DATA_DIR / "bytetrack_safety.yaml"


def ensure_tracker_cfg(process_fps: float) -> str:
    # track_buffer tính theo khung hình đã phân tích: giữ track mất dấu ~3 giây
    TRACKER_YAML.write_text(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.25\n"
        f"track_buffer: {max(10, int(3 * process_fps))}\nmatch_thresh: 0.8\nfuse_score: True\n",
        encoding="utf-8")
    return str(TRACKER_YAML)


class Detector:
    """Hai mô hình YOLO: pose (người + 17 keypoint) và detect (phương tiện)."""

    def __init__(self, s: Settings):
        from ultralytics import YOLO   # import muộn để kiểm thử logic không cần torch
        self.s = s
        self.pose = YOLO(s.pipeline.pose_model)
        self.det = YOLO(s.pipeline.det_model)
        self.tracker = ensure_tracker_cfg(s.pipeline.process_fps)

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        p = self.s.pipeline
        out: List[Detection] = []
        r = self.pose.track(frame, persist=True, tracker=self.tracker, conf=p.person_conf,
                            imgsz=p.imgsz, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for i in range(len(ids)):
                out.append(Detection(int(ids[i]), "person", "person", boxes[i], float(confs[i]),
                                     kp[i] if kp is not None else None))
        r = self.det.track(frame, persist=True, tracker=self.tracker, conf=p.vehicle_conf,
                           imgsz=p.imgsz, classes=list(VEHICLE_CLASSES), verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(ids)):
                out.append(Detection(int(ids[i]), "vehicle", VEHICLE_CLASSES.get(int(cls[i]), "vehicle"),
                                     boxes[i], float(confs[i])))
        return out


def open_source(source) -> cv2.VideoCapture:
    src = int(source) if isinstance(source, str) and source.isdigit() else source
    return cv2.VideoCapture(src)


class CameraWorker(threading.Thread):
    def __init__(self, cam: dict, hub: EventHub, settings: Settings = SETTINGS):
        super().__init__(daemon=True)
        self.cam, self.hub, self.s = cam, hub, settings
        self.stop_flag = threading.Event()
        self.frame_lock = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.raw: Optional[np.ndarray] = None
        self.status = "starting"
        self.error = ""
        self.fps = 0.0
        self.progress = 0.0
        self.engine: Optional[Engine] = None
        self.counts = {"persons": 0, "vehicles": 0}

    # ------------------------------------------------------------------
    def info(self) -> dict:
        d = {"status": self.status, "error": self.error, "fps": round(self.fps, 1),
             "progress": round(self.progress, 3), **self.counts}
        if self.engine:
            d["ground"] = self.engine.mapper.info()
        return d

    def apply_calibration(self) -> None:
        if self.engine is None:
            return
        cal = self.cam.get("calibration") or {}
        self.engine.set_calibration(cal.get("img_pts"), cal.get("ground_pts"))

    def _publish(self, img: np.ndarray) -> None:
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.frame_lock:
                self.jpeg = jpg.tobytes()
                self.frame_lock.notify_all()

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:   # hiển thị lỗi lên dashboard thay vì chết im lặng
            self.status, self.error = "error", f"{type(e).__name__}: {e}"

    def _run(self) -> None:
        src = self.cam["source"]
        is_file = self.cam.get("source_type") == "file"
        cap = open_source(src)
        if not cap.isOpened():
            self.status, self.error = "error", f"Không mở được nguồn video: {src}"
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if src_fps <= 1 or src_fps > 120:
            src_fps = 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        detector = Detector(self.s)
        self.status = "running"
        clips = ClipRecorder(self.s.pipeline.clip_pre, self.s.pipeline.clip_post)
        idx, wall0, last_t, t_hist = 0, time.monotonic(), -1e9, []
        while not self.stop_flag.is_set():
            step = max(1, int(round(src_fps / self.s.pipeline.process_fps))) if is_file else 1
            ok, frame = cap.read()
            if not ok:
                if is_file and self.cam.get("loop"):
                    cap.release()
                    cap = open_source(src)
                    idx, wall0 = 0, time.monotonic()
                    detector = Detector(self.s)
                    self.engine = None
                    continue
                self.status = "finished" if is_file else "error"
                if not is_file:
                    self.error = "Mất kết nối camera"
                break
            idx += 1
            if is_file:
                if (idx - 1) % step:
                    continue
                t = (idx - 1) / src_fps
                if n_frames:
                    self.progress = idx / n_frames
                if self.s.pipeline.realtime_files:
                    lag = t - (time.monotonic() - wall0)
                    if lag > 0:
                        time.sleep(lag)
            else:
                t = time.monotonic() - wall0
                if t - last_t < 1.0 / self.s.pipeline.process_fps:
                    continue
            last_t = t
            if self.engine is None:
                h, w = frame.shape[:2]
                self.engine = Engine(self.s, w, h)
                self.apply_calibration()
            self.raw = frame
            dets = detector(frame)
            events = self.engine.process(t, dets)
            self.counts = {"persons": sum(1 for x in self.engine.tm.persons() if not x.lost),
                           "vehicles": sum(1 for x in self.engine.tm.vehicles() if not x.lost)}
            img = draw(frame, self.engine, self.hub.recent_for(self.cam["id"]) +
                       [e["type"] for e in events if "type" in e], self.s.pipeline.show_bev)
            for ev in events:
                self.hub.emit(self.cam["id"], self.cam["name"], ev, img, clips)
            clips.push(t, img)
            self._publish(img)
            now = time.monotonic()
            t_hist = [x for x in t_hist if now - x < 2.0] + [now]
            self.fps = (len(t_hist) - 1) / max(t_hist[-1] - t_hist[0], 1e-3) if len(t_hist) > 1 else 0
        cap.release()
        if self.status == "running":
            self.status = "stopped"

    def stop(self) -> None:
        self.stop_flag.set()


# ---------------------------------------------------------------------------
def process_video(path: str, settings: Settings, on_frame: Optional[Callable] = None,
                  writer_path: Optional[str] = None) -> List[dict]:
    """Chạy không giao diện (dùng cho tools/evaluate.py và tools/run_video.py)."""
    cap = open_source(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / settings.pipeline.process_fps)))
    det = Detector(settings)
    eng, wr, events, idx, banner = None, None, [], 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % step:
            continue
        t = (idx - 1) / src_fps
        if eng is None:
            h, w = frame.shape[:2]
            eng = Engine(settings, w, h)
        evs = [e for e in eng.process(t, det(frame)) if "type" in e]
        events += evs
        if writer_path or on_frame:
            banner = [(tt, ty) for tt, ty in banner if t - tt <= 3.0] + [(t, e["type"]) for e in evs]
            img = draw(frame, eng, [ty for _, ty in banner], settings.pipeline.show_bev)
            if writer_path:
                if wr is None:
                    fps_out = src_fps / step
                    wr = cv2.VideoWriter(writer_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out,
                                         (img.shape[1], img.shape[0]))
                wr.write(img)
            if on_frame:
                on_frame(t, img, evs)
    cap.release()
    if wr:
        wr.release()
    return events
