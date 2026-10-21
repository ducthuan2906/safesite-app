"""Khởi động dashboard:  python run.py   rồi mở http://localhost:8000"""
import argparse

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"SafeSite đang chạy tại http://{a.host}:{a.port}")
    uvicorn.run("app.main:app", host=a.host, port=a.port, log_level="warning")
