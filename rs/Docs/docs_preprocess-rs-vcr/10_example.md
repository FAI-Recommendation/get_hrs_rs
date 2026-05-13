## Lý do còn 161 users (từ 711)

Cũng do **2 rule kết hợp** trong `process_data()`, nhưng tác động mạnh hơn với users:

### Tính toán thực tế

```
10.000 giao dịch ÷ 711 users = ~14 giao dịch/user (trung bình)

→ random_percent=0.5 → lấy 50%
→ trung bình còn ~7 giao dịch/user

→ n_core=10 → loại user có < 10 giao dịch
→ 7 < 10 → phần lớn users bị loại
→ chỉ còn những "heavy users" có rất nhiều giao dịch
→ 161 users sót lại
```

---

## Minh họa

```
User A: 20 giao dịch → sau 50% → 10 → đủ n_core=10 ✅ giữ lại
User B: 14 giao dịch → sau 50% →  7 → < n_core=10  ❌ bị loại
User C:  8 giao dịch → sau 50% →  4 → < n_core=10  ❌ bị loại
User D: 25 giao dịch → sau 50% → 12 → đủ n_core=10 ✅ giữ lại
```

Chỉ ~22% users (161/711) có đủ giao dịch để vượt qua cả 2 bộ lọc.

---

## Không phải mất dữ liệu — đây là rule cố ý

Đây là kỹ thuật chuẩn trong recommendation system gọi là **N-core filtering**:

| Mục đích | Giải thích |
|---|---|
| **Tránh cold-start** | User < 10 tương tác → model không học được gì về họ |
| **Đảm bảo chất lượng train/test** | Cần đủ giao dịch để chia 80/20 có nghĩa |
| **Sparse matrix** | User quá ít tương tác làm ma trận quá thưa → GCN hội tụ kém |

---

## Khuyến nghị với dataset 10k

`n_core=10` + `random_percent=0.5` **quá chặt** cho 10k records. Nên đổi:

```python
process_data(
    df,
    random_percent=1.0,  # dùng hết 100%
    n_core=3,            # hoặc n_core=5
    random_state=42
)
```

Kết quả dự kiến với `n_core=3, random_percent=1.0`:
- Users: ~400-500 (thay vì 161)
- Items: ~1800-2000 (thay vì 1475)