"""Download standard YOLOv3 COCO files. Requires internet access when run."""
from pathlib import Path
from urllib.request import urlretrieve

OUT = Path(__file__).resolve().parent / "models"
OUT.mkdir(exist_ok=True)
FILES = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
}
for name, url in FILES.items():
    target = OUT / name
    if target.exists() and target.stat().st_size > 1000:
        print(f"Already exists: {target}")
        continue
    print(f"Downloading {name} ...")
    try:
        urlretrieve(url, target)
        if target.stat().st_size < 1000:
            target.unlink(missing_ok=True)
            raise RuntimeError("Downloaded file is unexpectedly small; server may have returned an error page.")
        print(f"Saved {target} ({target.stat().st_size / (1024*1024):.1f} MB)")
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise SystemExit(f"Failed to download {name}: {exc}\nCheck your internet connection and retry.")
print("YOLOv3 files are ready. Run: streamlit run app.py")
