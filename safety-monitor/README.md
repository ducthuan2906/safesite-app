# SafeSite – Hệ thống thông minh phát hiện tai nạn lao động

Web dashboard phát hiện **té ngã**, **nguy cơ va chạm / va chạm người – phương tiện** và **va chạm phương tiện – phương tiện** từ video giám sát.
Môn Phát triển hệ thống thông minh – Nhóm: Nguyễn Đức Thuận, Nguyễn Đại Nghĩa, Nguyễn Minh Hiển.

## 1. Cài đặt

Yêu cầu Python 3.10+ (đã chạy thử với 3.13). Có GPU NVIDIA thì nhanh hơn, CPU vẫn chạy được (~5–10 FPS với mô hình `n`).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Máy chỉ có CPU, cài PyTorch bản CPU trước để nhẹ hơn nhiều:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

Lần chạy đầu, Ultralytics tự tải `yolo11n-pose.pt` và `yolo11n.pt` (~6 MB mỗi file).

## 2. Chạy dashboard

```bash
python run.py            # mở http://localhost:8000
```

1. Vào **Camera** → chọn *File video* / *Webcam* (nhập `0`) / *Camera IP* (`rtsp://...`) → **Thêm và chạy**.
2. (Nên làm) bấm **Hiệu chỉnh**: chọn 4 góc một hình chữ nhật trên nền, nhập kích thước thật (m). Không hiệu chỉnh thì hệ thống tự ước lượng tỉ lệ từ chiều cao người/xe (kém chính xác hơn, có ghi trên giao diện).
3. Tab **Giám sát** hiển thị luồng đã phân tích và cảnh báo realtime (âm thanh + thông báo trình duyệt).
4. Tab **Sự kiện**: xem ảnh + clip 6 giây, đánh dấu *Xác nhận / Đã xử lý / Báo nhầm*, xuất CSV.
5. Tab **Cài đặt**: chỉnh ngưỡng, có hiệu lực ngay.

## 3. Các loại sự kiện

| Mã | Mức | Điều kiện chính |
|---|---|---|
| `FALL` | Nghiêm trọng | Rơi nhanh (hông ≥ 0,8 chiều cao/giây hoặc chiều cao giảm ≥ 35% trong 0,7 s) → nằm (góc thân ≥ 60°) trong ≤ 2 s → giữ ≥ 1 s |
| `FALL_IMMOBILE` | Nghiêm trọng | Sau FALL, bất động > 10 s |
| `PERSON_DOWN` | Cảnh báo | Nằm bất động ≥ 5 s nhưng không thấy pha rơi (ngã từ từ / mất pha rơi) |
| `NEAR_MISS` | Cảnh báo | Xe đang chạy cách người < 2 m, hoặc TTC < 2 s |
| `PERSON_VEHICLE_COLLISION` | Nghiêm trọng | Tiếp xúc (< 0,4 m trên nền + bbox giao nhau) **và** trong 2 s người ngã / bị đẩy / mất dấu; hoặc FALL ≤ 3 s sau khi ở sát xe đang chạy |
| `VEHICLE_COLLISION` | Nghiêm trọng | Footprint hai xe chạm nhau, trước đó tiến lại gần > 1 m/s, sau đó một xe giảm tốc > 50% hoặc đổi hướng > 30° |

## 4. Kiểm thử và đánh giá

```bash
python -m pytest -q tests                     # 17 kiểm thử: 16 kịch bản giả lập + chấm điểm, không cần GPU/video
python -m tools.run_video video.mp4 --out out.mp4
python -m tools.evaluate --labels data/labels.csv --videos data/test_videos --out data/eval
```

`labels.csv` (một dòng cho mỗi sự kiện; video không có sự kiện ghi `NONE`):

```csv
video,type,start,end
fall_01.mp4,FALL,4.2,5.6
kneel_03.mp4,NONE,0,0
cart_02.mp4,NEAR_MISS,7.0,8.5
```

Tạo nhanh video bán giả lập (cắt người thật đang đứng, xoay quanh bàn chân để mô phỏng ngã) khi chưa có dữ liệu – chỉ để kiểm tra luồng xử lý, không thay thế dữ liệu thật:

```bash
python -m tools.make_semisynthetic src.mp4 --frame 50 --box 472,80,556,260 --bg 100 --out fall_synth.mp4
```

Đánh giá theo **sự kiện**: cảnh báo cùng loại trong `[start − 1 s, end + 3 s]` là TP. Báo cáo Precision, Recall, F1, độ trễ trung bình, FP/giờ.
Để làm thí nghiệm so sánh A–D, chạy lại với các cấu hình khác nhau và lưu `report.json` của từng lần.

## 5. Cấu trúc mã

```
app/
  config.py        ngưỡng mặc định (dataclass) – lưu ở data/settings.json
  tracks.py        Track, lịch sử, vận tốc trên nền; nối lại ID
  geometry.py      homography / tự hiệu chỉnh từ chiều cao vật, đa giác footprint
  analyzers/fall.py       máy trạng thái té ngã trên pose
  analyzers/collision.py  người–xe, xe–xe, TTC, near-miss, xác nhận va chạm
  engine.py        lõi phân tích 1 camera (không phụ thuộc mô hình → test được)
  pipeline.py      đọc video, YOLO + ByteTrack, luồng xử lý mỗi camera
  annotate.py      vẽ khung xương, bbox, khoảng cách, bản đồ mặt nền
  events.py        SQLite, ảnh chụp, clip .webm, WebSocket hub
  main.py          FastAPI: REST, WebSocket, MJPEG
static/            giao diện (HTML/CSS/JS thuần, không cần build)
tools/             evaluate.py, run_video.py
tests/             bộ giả lập + kịch bản kiểm thử
```

## 6. Nhận diện xe nâng, máy xúc (fine-tune)

COCO **không có** lớp forklift/excavator; bản mặc định dùng car/truck/bus/motorcycle/bicycle. Để dùng trong kho/công trường:

```bash
yolo detect train model=yolo11n.pt data=construction.yaml epochs=50 imgsz=640
```

Sau đó đặt đường dẫn `best.pt` vào *Cài đặt → Mô hình phát hiện xe* và sửa `VEHICLE_CLASSES`, `CLASS_HEIGHT_M` trong `app/config.py` theo tên lớp mới.

## 7. Giới hạn đã biết

- Mô hình COCO nhận diện kém xe nhìn thẳng từ trên xuống (camera trần, drone): khi thử, ô tô nhìn từ trên bị nhận thành “cell phone”. Cần camera nhìn chéo hoặc fine-tune.
- Footprint xe xấp xỉ bằng 35% phần dưới bbox: đúng nhất khi camera gắn cao, nhìn chéo xuống.
- Homography giả định nền phẳng; dốc, bậc thang, giàn giáo sẽ đo sai.
- Ngưỡng mặc định chưa được hiệu chỉnh trên dữ liệu thật – cần chạy `tools/evaluate.py` trên bộ kiểm thử của nhóm.
- Hệ thống chỉ hỗ trợ người giám sát, không thay thế quy trình an toàn lao động.
