"""
preprocess_vcr_clip.py
======================
Phiên bản dùng CLIP embedding (512-D) thay cho MobileNetV2 (1280-D → PCA 768).

Thay đổi so với preprocess_vcr_10k.py:
  - EMBEDDINGS_DIR  : embeddings_clip_10k/   (file .npy shape (512,) từ CLIP)
  - OUTPUT (ROOT)   : clip_10k_sample/        (tất cả output vào đây)
  - feature3        : 512-D trực tiếp, KHÔNG cần PCA (CLIP đã compact + L2-normalized)
  - Mọi thứ khác giữ nguyên (clean, n-core, ID mapping, BERT text embedding)

Output directory: clip_10k_sample/
"""

import ast
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = r"E:\DoCode\CD2\source\Source\get_hrs_rs\rs\get10k_data"

# Input: vẫn đọc CSV từ output_10k_sample/preprocess/
ROOT_PRESS = os.path.join(BASE_DIR, "output_10k_sample", "preprocess")

# Input: CLIP embeddings
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "output_10k_sample", "embeddings_clip_10k")

# Output: thư mục riêng cho pipeline CLIP
OUTPUT = os.path.join(BASE_DIR, "clip_10k_sample")
os.makedirs(OUTPUT, exist_ok=True)

# process_data params
RANDOM_PERCENT = 1.0
N_CORE         = 5
RANDOM_STATE   = 42

CLIP_DIM = 512   # CLIP ViT-B/32 output dimension (không cần PCA)

print(f"📁 Output → {OUTPUT}")
print(f"🖼️  Embeddings → {EMBEDDINGS_DIR}")


# ─────────────────────────────────────────────
# STEP 1 – Load & clean outfits
# ─────────────────────────────────────────────
print("\n📂 STEP 1 – Load outfits")
outfits = pd.read_csv(os.path.join(ROOT_PRESS, "outfits_10k.csv"), sep=";", on_bad_lines="skip")
outfits["outfit_tags"]    = outfits["outfit_tags"].apply(ast.literal_eval)
outfits["tag_categories"] = outfits["tag_categories"].apply(ast.literal_eval)

null_outfit_ids = outfits[outfits["name"].isnull()]["id"]
outfits = outfits.dropna(subset=["name"])
outfits["description"] = outfits["description"].fillna("No description available")
outfits.reset_index(drop=True, inplace=True)
print(f"   outfits: {len(outfits)} rows  (dropped {len(null_outfit_ids)} null-name)")


# ─────────────────────────────────────────────
# STEP 2 – Load & clean pictures
# ─────────────────────────────────────────────
print("\n📂 STEP 2 – Load pictures")
pictures = pd.read_csv(os.path.join(ROOT_PRESS, "picture_triplets_10k.csv"), sep=";", on_bad_lines="skip")
pictures = pictures[~pictures["outfit.id"].isin(null_outfit_ids)]
print(f"   pictures: {len(pictures)} rows  |  unique outfits: {pictures['outfit.id'].nunique()}")


# ─────────────────────────────────────────────
# STEP 3 – Load & clean transactions
# ─────────────────────────────────────────────
print("\n📂 STEP 3 – Load transactions")
transactions = pd.read_csv(os.path.join(ROOT_PRESS, "user_activity_triplets_10k.csv"), sep=";", on_bad_lines="skip")
transactions["rentalPeriod.start"] = pd.to_datetime(transactions["rentalPeriod.start"])
transactions["rentalPeriod.end"]   = pd.to_datetime(transactions["rentalPeriod.end"])
transactions = transactions[~transactions["outfit.id"].isin(null_outfit_ids)]
transactions = transactions[transactions["outfit.id"].isin(pictures["outfit.id"])]
print(f"   transactions: {len(transactions)} rows  |  users: {transactions['customer.id'].nunique()}  |  outfits: {transactions['outfit.id'].nunique()}")


# ─────────────────────────────────────────────
# STEP 4 – Tạo dataset_VCR.csv
# ─────────────────────────────────────────────
print("\n💾 STEP 4 – Tạo dataset_VCR.csv")
data_collect = transactions.rename(columns={
    "outfit.id":          "item_id_original",
    "customer.id":        "user_id_original",
    "rentalPeriod.start": "time",
})
data_collect["time"] = pd.to_datetime(data_collect["time"])
data_collect["time"] = data_collect["time"].apply(lambda x: int(time.mktime(x.timetuple())))
data_collect = data_collect[["user_id_original", "item_id_original", "time"]]
data_collect.to_csv(os.path.join(OUTPUT, "dataset_VCR.csv"), index=False)
print(f"   Saved dataset_VCR.csv  |  {len(data_collect)} records")


# ─────────────────────────────────────────────
# STEP 5 – process_data (n-core + ID mapping)
# ─────────────────────────────────────────────
print(f"\n⚙️  STEP 5 – process_data  (random={RANDOM_PERCENT}, n_core={N_CORE})")
df = pd.read_csv(os.path.join(OUTPUT, "dataset_VCR.csv"))

np.random.seed(RANDOM_STATE)
n_samples  = int(len(df) * RANDOM_PERCENT)
sampled_df = df.sample(n=n_samples, random_state=RANDOM_STATE)

user_counts = sampled_df["user_id_original"].value_counts()
valid_users = user_counts[user_counts >= N_CORE].index
filtered_df = sampled_df[sampled_df["user_id_original"].isin(valid_users)]

user_id_map = {old: new for new, old in enumerate(filtered_df["user_id_original"].unique())}
item_id_map = {old: new for new, old in enumerate(filtered_df["item_id_original"].unique())}

mapped_df = filtered_df.copy()
mapped_df["user_id"] = mapped_df["user_id_original"].map(user_id_map)
mapped_df["item_id"] = mapped_df["item_id_original"].map(item_id_map)

out_path = os.path.join(OUTPUT, f"dataset_VCR_{RANDOM_PERCENT}_{RANDOM_STATE}_{N_CORE}.csv")
mapped_df.to_csv(out_path, index=False)
print(f"   Users: {mapped_df['user_id'].nunique()}  |  Items: {mapped_df['item_id'].nunique()}  |  Records: {len(mapped_df)}")
print(f"   Saved: {out_path}")

df = pd.read_csv(out_path)
df = df.sort_values(by=["user_id", "time"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 6 – Sinh các file mapping / interaction
# ─────────────────────────────────────────────
print("\n💾 STEP 6 – Sinh user_list / item_list / intersection_user / train / test")

user_df = df[["user_id_original", "user_id"]].drop_duplicates().sort_values("user_id")
user_df.to_csv(os.path.join(OUTPUT, "user_list.txt"), sep=" ", index=False, header=False)

item_df = df[["item_id_original", "item_id"]].drop_duplicates().sort_values("item_id").reset_index(drop=True)
item_df.to_csv(os.path.join(OUTPUT, "item_list.txt"), sep=" ", index=False, header=False)

user_item_interactions = df.groupby("user_id")["item_id"].apply(lambda x: sorted(x)).to_dict()
with open(os.path.join(OUTPUT, "intersection_user.txt"), "w") as f:
    for uid in sorted(user_item_interactions.keys()):
        f.write(f"{uid} {' '.join(map(str, user_item_interactions[uid]))}\n")

df["rank"] = df.groupby("user_id").cumcount() + 1
user_counts2 = df.groupby("user_id")["rank"].max().reset_index()
user_counts2.rename(columns={"rank": "total_interactions"}, inplace=True)
user_counts2["train_threshold"] = (user_counts2["total_interactions"] * 0.8).astype(int)
df = df.merge(user_counts2, on="user_id", how="inner")

train_interactions, test_interactions = {}, {}
for user_id, udf in df.groupby("user_id"):
    th = udf["train_threshold"].iloc[0]
    train_interactions[user_id] = udf[udf["rank"] <= th]["item_id"].tolist()
    test_interactions[user_id]  = udf[udf["rank"] >  th]["item_id"].tolist()

with open(os.path.join(OUTPUT, "train.txt"), "w") as f:
    for uid, items in train_interactions.items():
        f.write(f"{uid} {' '.join(map(str, items))}\n")

with open(os.path.join(OUTPUT, "test.txt"), "w") as f:
    for uid, items in test_interactions.items():
        f.write(f"{uid} {' '.join(map(str, items))}\n")

print(f"   user_list.txt  : {len(user_df)} users")
print(f"   item_list.txt  : {len(item_df)} items")
print(f"   train/test split: 80/20 per user")


# ─────────────────────────────────────────────
# STEP 7 – Build items_features.csv (CLIP)
# ─────────────────────────────────────────────
print("\n🔨 STEP 7 – Build items_features.csv  (CLIP 512-D, no PCA)")

merged_df = pd.merge(outfits, item_df, left_on="id", right_on="item_id_original", how="inner")

grouped_pictures = (
    pictures.sort_values(by=["outfit.id", "displayOrder"])
    .groupby("outfit.id", as_index=False)
    .agg({
        "picture.id":   lambda x: list(x),
        "file_name":    lambda x: list(x),
        "displayOrder": lambda x: list(x),
    })
)

items_features_df = pd.merge(
    merged_df, grouped_pictures,
    left_on="item_id_original", right_on="outfit.id",
    how="inner",
)

items_features_df["outfit_tags"] = items_features_df["outfit_tags"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else x
)
items_features_df["feature1"] = items_features_df["name"] + " " + items_features_df["outfit_tags"]
items_features_df["feature2"] = items_features_df["description"]

# image_list.txt
image_list_df = (
    items_features_df[["item_id_original", "item_id", "file_name"]]
    .sort_values("item_id")
    .reset_index(drop=True)
)
image_list_df.to_csv(os.path.join(OUTPUT, "image_list.txt"), sep=" ", index=False, header=False)

# ── feature3: load CLIP .npy (512-D, đã L2-normalize sẵn) ─────────────
print(f"   Loading CLIP embeddings from {EMBEDDINGS_DIR} ...")

def create_embedding_paths(picture_ids):
    return [
        os.path.join(EMBEDDINGS_DIR, f"{pid.split('.')[1]}.npy")
        for pid in picture_ids
    ]

def load_mean_embeddings_clip(paths):
    """Mean của các CLIP vector (thường chỉ có 1 ảnh/outfit trong dataset 10k)."""
    vecs = []
    for p in paths:
        if os.path.isfile(p):
            vecs.append(np.load(p).astype(np.float32).flatten())
    if vecs:
        # Re-normalize sau mean để giữ unit norm
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        return (mean_vec / norm if norm > 0 else mean_vec).tolist()
    return [0.0] * CLIP_DIM

items_features_df["embedding_paths"] = items_features_df["picture.id"].apply(create_embedding_paths)
items_features_df["feature3"]        = items_features_df["embedding_paths"].apply(load_mean_embeddings_clip)

# ── Kiểm tra chiều vector ─────────────────────────────────────────────
sample_len = len(items_features_df["feature3"].iloc[0])
print(f"   feature3 dim: {sample_len}  (expected {CLIP_DIM})")
assert sample_len == CLIP_DIM, f"❌ Sai chiều! Kiểm tra lại embeddings_clip_10k/"

# ── Lưu items_features.csv ────────────────────────────────────────────
fe_df = items_features_df[["item_id", "feature1", "feature2", "feature3"]].copy()
fe_df.to_csv(os.path.join(OUTPUT, "items_features.csv"), index=False)
print(f"   Saved items_features.csv  |  {len(fe_df)} items  |  feature3={CLIP_DIM}-D")


# ─────────────────────────────────────────────
# STEP 8 – BERT text embeddings
# ─────────────────────────────────────────────
print("\n🤖 STEP 8 – BERT text embeddings")
xxx = pd.read_csv(os.path.join(OUTPUT, "items_features.csv"))

xxx["combined_text"]         = xxx["feature1"] + " " + xxx["feature2"]
xxx["cleaned_combined_text"] = xxx["combined_text"].apply(str.lower)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

print("   Loading bert-base-uncased ...")
tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
bert_model.eval()

BERT_BATCH = 32

def get_bert_embeddings_batch(texts: list) -> np.ndarray:
    inputs = tokenizer(texts, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # (N, 768)

texts = xxx["cleaned_combined_text"].tolist()
print(f"   Generating embeddings for {len(texts)} items  (batch={BERT_BATCH}) ...")

all_embeddings = []
for i in tqdm(range(0, len(texts), BERT_BATCH)):
    all_embeddings.append(get_bert_embeddings_batch(texts[i : i + BERT_BATCH]))

text_embeddings = np.vstack(all_embeddings)

def parse_vector_string(s):
    # Hỗ trợ cả "0.1 0.2 0.3" (space) lẫn "0.1, 0.2, 0.3" (comma)
    return np.array([float(x) for x in s.strip("[]").replace(",", " ").split()], dtype=np.float32)

image_embeddings = np.vstack(xxx["feature3"].apply(parse_vector_string).values)

text_embeddings  = text_embeddings.astype(np.float32)
image_embeddings = image_embeddings.astype(np.float32)

print(f"   text_embeddings  shape: {text_embeddings.shape}")
print(f"   image_embeddings shape: {image_embeddings.shape}")

np.save(os.path.join(OUTPUT, "text_embeddings.npy"),  text_embeddings)
np.save(os.path.join(OUTPUT, "image_embeddings.npy"), image_embeddings)
print(f"   Saved text_embeddings.npy & image_embeddings.npy")

print("\n✅ Preprocessing hoàn tất!")
print(f"   📁 Tất cả output tại: {OUTPUT}")
