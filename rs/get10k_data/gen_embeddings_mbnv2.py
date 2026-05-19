"""
Extract image_embeddings.npy và text_embeddings.npy cho output_10k_sample.

image_embeddings: lấy từ feature3 trong items_features.csv (MobileNetV2 PCA-768D)
text_embeddings:  re-run BERT trên feature1 + feature2

Chạy từ repo root:
    python3 rs/get10k_data/gen_embeddings_mbnv2.py
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

ROOT = os.path.join(os.path.dirname(__file__), "output_10k_sample")

# ── Load items_features.csv ──────────────────────────────────────────────
print("📂 Loading items_features.csv ...")
df = pd.read_csv(os.path.join(ROOT, "items_features.csv"))
print(f"   {len(df)} items")

# ── Image embeddings từ feature3 ────────────────────────────────────────
print("\n🖼️  Extracting image embeddings from feature3 ...")

def parse_vector(s):
    return np.array([float(x) for x in str(s).strip("[]").split()], dtype=np.float32)

image_embeddings = np.vstack(df["feature3"].apply(parse_vector).values)
image_embeddings = image_embeddings.astype(np.float32)
print(f"   image_embeddings shape: {image_embeddings.shape}")

np.save(os.path.join(ROOT, "image_embeddings.npy"), image_embeddings)
print(f"   ✅ Saved image_embeddings.npy")

# ── Text embeddings via BERT ──────────────────────────────────────────────
txt_path = os.path.join(ROOT, "text_embeddings.npy")
if os.path.exists(txt_path):
    print(f"\n📝 text_embeddings.npy already exists — skip BERT")
else:
    print("\n🤖 Running BERT for text embeddings ...")
    from transformers import AutoTokenizer, AutoModel

    BERT_MODEL = "bert-base-uncased"
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL).to(device).eval()

    df["combined_text"] = (df["feature1"].fillna("") + " " + df["feature2"].fillna("")).str.lower()
    texts = df["combined_text"].tolist()

    def get_bert_batch(batch):
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        return out.last_hidden_state[:, 0, :].cpu().numpy()

    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        if i % 256 == 0:
            print(f"   BERT {i}/{len(texts)} ...")
        all_embs.append(get_bert_batch(texts[i : i + BATCH_SIZE]))

    text_embeddings = np.vstack(all_embs).astype(np.float32)
    print(f"   text_embeddings shape: {text_embeddings.shape}")

    np.save(txt_path, text_embeddings)
    print(f"   ✅ Saved text_embeddings.npy")

print("\n✅ Done! Files saved to:", ROOT)
print(f"   image_embeddings.npy: {np.load(os.path.join(ROOT, 'image_embeddings.npy')).shape}")
print(f"   text_embeddings.npy:  {np.load(txt_path).shape}")
