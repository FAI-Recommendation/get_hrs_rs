"""
WORKFLOW - Tạo 10k sample từ VCR dataset
=========================================

INPUT (3 file gốc trong preprocess/):
  - user_activity_triplets.csv  : giao dịch thuê đồ  (customer.id, outfit.id, rentalPeriod.start, rentalPeriod.end)
  - picture_triplets.csv        : ảnh của outfit      (picture.id, outfit.id, displayOrder, file_name)
  - outfits.csv                 : thông tin outfit     (id, name, description, ...)

OUTPUT (3 file trong output_10k_sample/):
  - user_activity_triplets_10k.csv  : ~10.000 giao dịch  (giữ nguyên cột gốc)
  - picture_triplets_10k.csv        : 1 ảnh / outfit      (giữ nguyên cột gốc, chỉ displayOrder==0)
  - outfits_10k.csv                 : outfit tương ứng    (giữ nguyên cột gốc)

RÀNG BUỘC:
  - Tên file output = tên file gốc + "_10k" (không thêm/xóa cột)
  - 1 outfit = 1 ảnh đại diện (displayOrder == 0, không trùng outfit.id)
  - outfit count == picture count (bảo đảm tính nhất quán)

GIẢI THUẬT (5 bước):

  Bước 1 - pick_main_picture_rows()
    picture_triplets.csv (50k rows, nhiều ảnh/outfit)
    → lọc displayOrder == 0
    → drop_duplicates(outfit.id)         ← đảm bảo 1 outfit = 1 ảnh
    → pictures_main (~N outfit)

  Bước 2 - merge_sources()
    Chỉ giữ lại các giao dịch mà outfit CÓ ảnh chính VÀ có trong outfits.csv
    interactions INNER JOIN pictures_main ON outfit.id
                 INNER JOIN outfits        ON outfit.id = id
    → merged (giao dịch hợp lệ, đầy đủ thông tin)

  Bước 3 - sample_by_users()
    Từ merged, chọn user có >= 3 giao dịch
    Cộng dồn toàn bộ giao dịch của từng user (shuffle ngẫu nhiên) cho đến khi >= 10.000
    → sampled (~10.000 rows)

  Bước 4 - Truy ngược về interactions_orig
    sampled chứa _orig_row_id → lấy lại đúng dòng gốc từ interactions_orig
    → sampled_interactions (giữ nguyên 100% cột gốc, bỏ cột tạm _orig_row_id)

  Bước 5 - build_outputs()
    Từ sampled_interactions lấy tập outfit.id đã chọn
    pictures_sampled = pictures_main  WHERE outfit.id IN sampled_outfit_ids  → đúng N ảnh
    outfits_sampled  = outfits_orig   WHERE id         IN sampled_outfit_ids  → đúng N outfit
    → outfit count == picture count == N
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep=";", on_bad_lines="skip")


def pick_main_picture_rows(pictures: pd.DataFrame) -> pd.DataFrame:
    """Bước 1: Lọc ảnh chính - 1 outfit = 1 ảnh (displayOrder == 0)."""
    required_cols = {"picture.id", "outfit.id", "displayOrder", "file_name"}
    missing = required_cols - set(pictures.columns)
    if missing:
        raise ValueError(f"picture_triplets.csv is missing columns: {sorted(missing)}")

    main_pictures = pictures[pictures["displayOrder"] == 0].copy()
    main_pictures = main_pictures.drop_duplicates(subset=["outfit.id"], keep="first")
    return main_pictures


def merge_sources(
    interactions: pd.DataFrame,
    pictures_main: pd.DataFrame,
    outfits: pd.DataFrame,
) -> pd.DataFrame:
    """Bước 2: Chỉ giữ giao dịch có outfit hợp lệ (có ảnh chính + có trong outfits.csv)."""
    required_interaction_cols = {"customer.id", "outfit.id", "rentalPeriod.start", "rentalPeriod.end"}
    required_outfit_cols = {"id", "name", "description"}

    missing_interactions = required_interaction_cols - set(interactions.columns)
    missing_outfits = required_outfit_cols - set(outfits.columns)

    if missing_interactions:
        raise ValueError(f"user_activity_triplets.csv is missing columns: {sorted(missing_interactions)}")
    if missing_outfits:
        raise ValueError(f"outfits.csv is missing columns: {sorted(missing_outfits)}")

    interactions = interactions.copy()
    outfits = outfits.copy()
    pictures_main = pictures_main.copy()

    interactions["rentalPeriod.start"] = pd.to_datetime(interactions["rentalPeriod.start"], errors="coerce")
    interactions["rentalPeriod.end"] = pd.to_datetime(interactions["rentalPeriod.end"], errors="coerce")
    interactions = interactions.dropna(subset=["customer.id", "outfit.id", "rentalPeriod.start"])

    outfits = outfits.dropna(subset=["id", "name"])
    outfits["description"] = outfits["description"].fillna("No description available")

    merged = interactions.merge(pictures_main, on="outfit.id", how="inner")
    merged = merged.merge(outfits, left_on="outfit.id", right_on="id", how="inner")

    return merged


def sample_by_users(
    df: pd.DataFrame,
    target_records: int = 10_000,
    min_user_interactions: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bước 3: Chọn user có >= min_user_interactions, cộng dồn đến khi đủ target_records."""
    if df.empty:
        raise ValueError("Input dataframe is empty after merging.")

    user_counts = df["customer.id"].value_counts()
    eligible_users = user_counts[user_counts >= min_user_interactions].index.to_list()

    if not eligible_users:
        raise ValueError(
            "No users satisfy the minimum interaction threshold. "
            f"Try lowering min_user_interactions below {min_user_interactions}."
        )

    shuffled_users = pd.Series(eligible_users).sample(frac=1.0, random_state=random_state).tolist()

    selected_frames: List[pd.DataFrame] = []
    total_records = 0

    for user_id in shuffled_users:
        user_rows = df[df["customer.id"] == user_id].sort_values("rentalPeriod.start")
        selected_frames.append(user_rows)
        total_records += len(user_rows)
        if total_records >= target_records:
            break

    sampled = pd.concat(selected_frames, ignore_index=True)

    if len(sampled) > target_records:
        sampled = sampled.sort_values(["customer.id", "rentalPeriod.start"]).head(target_records).reset_index(drop=True)

    sampled = sampled.sort_values(["customer.id", "rentalPeriod.start"]).reset_index(drop=True)
    return sampled


def build_outputs(
    sampled_interactions: pd.DataFrame,
    pictures_main: pd.DataFrame,
    outfits_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bước 5: Lấy pictures và outfits tương ứng với outfit.id trong sample.
    Đảm bảo: outfit count == picture count (1 outfit = 1 ảnh chính).
    """
    sampled_outfit_ids = set(sampled_interactions["outfit.id"].unique())

    # pictures_main đã là displayOrder==0 + dedup → 1 outfit = 1 ảnh
    pictures_sampled = pictures_main[pictures_main["outfit.id"].isin(sampled_outfit_ids)].copy()
    outfits_sampled = outfits_all[outfits_all["id"].isin(sampled_outfit_ids)].copy()

    return sampled_interactions, pictures_sampled, outfits_sampled


def save_outputs(
    interactions: pd.DataFrame,
    pictures: pd.DataFrame,
    outfits: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Lưu output: tên file gốc + _10k, giữ nguyên cột gốc, sep=';'."""
    output_dir.mkdir(parents=True, exist_ok=True)

    interactions.to_csv(output_dir / "user_activity_triplets_10k.csv", index=False, sep=";")
    pictures.to_csv(output_dir / "picture_triplets_10k.csv", index=False, sep=";")
    outfits.to_csv(output_dir / "outfits_10k.csv", index=False, sep=";")


def print_summary(
    sampled: pd.DataFrame,
    pictures_out: pd.DataFrame,
    outfits_out: pd.DataFrame,
) -> None:
    print("--- 10k sample summary ---")
    print(f"Records (giao dịch)  : {len(sampled)}")
    print(f"Users                : {sampled['customer.id'].nunique()}")
    print(f"Outfits              : {outfits_out['id'].nunique()}")
    print(f"Pictures (main only) : {len(pictures_out)}  ← phải == Outfits")
    if outfits_out["id"].nunique() != len(pictures_out):
        print("⚠️  WARNING: outfit count != picture count!")
    else:
        print("✅  outfit count == picture count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a ~10k user-centric sample from the VCR preprocess CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "preprocess",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output_10k_sample",
    )
    parser.add_argument("--target-records", type=int, default=10_000)
    parser.add_argument("--min-user-interactions", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load ---
    print("--- Loading source files ---")
    interactions = read_csv(args.input_dir / "user_activity_triplets.csv")
    pictures     = read_csv(args.input_dir / "picture_triplets.csv")
    outfits      = read_csv(args.input_dir / "outfits.csv")

    # Đánh số dòng tạm để truy ngược về interactions gốc sau khi merge (bước 4)
    interactions["_orig_row_id"] = range(len(interactions))
    interactions_orig = interactions.copy()
    outfits_orig      = outfits.copy()

    # --- Bước 1: lọc ảnh chính ---
    print("--- Bước 1: Lọc ảnh chính (displayOrder == 0, 1 outfit = 1 ảnh) ---")
    pictures_main = pick_main_picture_rows(pictures)
    print(f"  pictures_main: {len(pictures_main)} rows")

    # --- Bước 2: merge để loại outfit không hợp lệ ---
    print("--- Bước 2: Merge interactions + pictures_main + outfits ---")
    merged = merge_sources(interactions, pictures_main, outfits)
    print(f"  merged: {len(merged)} rows")

    # --- Bước 3: sample ~10k giao dịch ---
    print("--- Bước 3: Sample ~10k giao dịch theo user ---")
    sampled = sample_by_users(
        merged,
        target_records=args.target_records,
        min_user_interactions=args.min_user_interactions,
        random_state=args.random_state,
    )
    print(f"  sampled: {len(sampled)} rows")

    # --- Bước 4: truy ngược về dòng gốc (giữ nguyên 100% cột interactions) ---
    print("--- Bước 4: Truy ngược về interactions gốc ---")
    sampled_orig_ids = sampled["_orig_row_id"].unique().tolist()
    sampled_interactions = (
        interactions_orig[interactions_orig["_orig_row_id"].isin(sampled_orig_ids)]
        .drop_duplicates()
        .drop(columns=["_orig_row_id"])
    )

    # --- Bước 5: build output pictures + outfits tương ứng ---
    print("--- Bước 5: Lấy pictures và outfits tương ứng ---")
    interactions_out, pictures_out, outfits_out = build_outputs(
        sampled_interactions, pictures_main, outfits_orig
    )

    # --- Save ---
    print("--- Saving outputs ---")
    save_outputs(interactions_out, pictures_out, outfits_out, args.output_dir)
    print_summary(sampled, pictures_out, outfits_out)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
