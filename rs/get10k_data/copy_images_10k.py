"""
Copy 2223 main images (displayOrder == 0) của 10k sample
từ images gốc sang thư mục images_10k.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


PICTURES_10K = Path(__file__).resolve().parent / "output_10k_sample" / "picture_triplets_10k.csv"
SOURCE_DIR   = Path(__file__).resolve().parent.parent / "detect_YOLOv5s" / "images"
OUTPUT_DIR   = Path(__file__).resolve().parent / "output_10k_sample" / "images_2223_main"


def main() -> None:
    df = pd.read_csv(PICTURES_10K, sep=";")
    # picture_triplets_10k.csv đã chỉ chứa displayOrder == 0 (fix trong slipt_10k_sample.py)
    main_pics = df["file_name"].dropna().unique().tolist()
    print(f"Cần copy: {len(main_pics)} ảnh")

    # Xóa folder cũ để tránh còn sót ảnh từ lần chạy trước
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    copied, missing = 0, []
    for fname in main_pics:
        src = SOURCE_DIR / fname
        dst = OUTPUT_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(fname)

    print(f"✅ Đã copy: {copied} ảnh → {OUTPUT_DIR}")
    if missing:
        print(f"⚠️  Không tìm thấy {len(missing)} ảnh:")
        for f in missing[:10]:
            print(f"   {f}")


if __name__ == "__main__":
    main()
