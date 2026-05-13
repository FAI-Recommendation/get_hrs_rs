# 07 – Tạo các file danh sách: user_list, item_list, intersection_user, image_list

## Mục tiêu

Sau khi đã có bảng tương tác đã mapping ID (`dataset_VCR_0.5_42_10.csv`), bước này tạo ra 4 file phụ trợ mà `load_data.py` và các model LightGCN/NGCF cần đọc để dựng đồ thị.

---

## Input

| Biến | Nguồn |
|---|---|
| `user_df` | DataFrame chứa `user_id_original`, `user_id` |
| `item_df` | DataFrame chứa `item_id_original`, `item_id` |
| `user_item_interactions` | dict `{user_id: [item_id, ...]}` từ bảng tương tác |
| `items_features_df` | DataFrame đã merge outfit + picture (có `file_name`) |

---

## 1. `user_list.txt`

**Mục đích:** Ánh xạ giữa ID gốc (`user_id_original`) và ID số liên tiếp (`user_id`).  
Model cần ID số liên tiếp để tạo embedding matrix đúng kích thước.

```python
user_list_df = user_df.sort_values(by='user_id')
user_list_df.to_csv('user_list.txt', sep=' ', index=False, header=False)
```

**Định dạng file:**
```
customer.123  0
customer.456  1
customer.789  2
...
```

Mỗi dòng: `user_id_original  user_id` (cách nhau bởi dấu cách)

---

## 2. `item_list.txt`

**Mục đích:** Ánh xạ giữa `item_id_original` (outfit.id gốc dạng string) và `item_id` (số nguyên liên tiếp).

```python
item_list_df = item_df.sort_values(by='item_id')
item_list_df.to_csv('item_list.txt', sep=' ', index=False, header=False)
```

**Định dạng file:**
```
outfit.abc123  0
outfit.def456  1
outfit.ghi789  2
...
```

---

## 3. `intersection_user.txt`

**Mục đích:** Liệt kê toàn bộ item mà mỗi user đã tương tác (train + test gộp lại).  
Dùng để dựng ma trận tương tác `R` trong `load_data.py`.

```python
user_item_interactions = {}
for _, row in df.iterrows():
    uid = row['user_id']
    iid = row['item_id']
    if uid not in user_item_interactions:
        user_item_interactions[uid] = []
    user_item_interactions[uid].append(iid)

with open('intersection_user.txt', 'w') as f:
    for user_id in sorted(user_item_interactions.keys()):
        item_list_str = ' '.join(map(str, user_item_interactions[user_id]))
        f.write(f"{user_id} {item_list_str}\n")
```

**Định dạng file:**
```
0 1 5 8 12 23
1 2 6 9 15
2 0 3 7 11 19 25
...
```

Mỗi dòng: `user_id item_id1 item_id2 ...`

---

## 4. `image_list.txt`

**Mục đích:** Ánh xạ `item_id_original`, `item_id` và `file_name` ảnh tương ứng.  
Notebook dùng file này để biết ảnh nào thuộc item nào khi load embedding.

```python
image_list_df = items_features_df[['item_id_original', 'item_id', 'file_name']]
image_list_df = image_list_df.sort_values(by='item_id')
image_list_df.reset_index(drop=True, inplace=True)
image_list_df.to_csv('image_list.txt', sep=' ', index=False, header=False)
```

**Định dạng file:**
```
outfit.abc123  0  abc123.jpg
outfit.def456  1  def456.jpg
...
```

---

## Output tổng kết

| File | Cột | Vai trò trong model |
|---|---|---|
| `user_list.txt` | `user_id_original  user_id` | Mapping để debug, không dùng trực tiếp trong GCN |
| `item_list.txt` | `item_id_original  item_id` | Mapping để debug |
| `intersection_user.txt` | `user_id item_id1 item_id2...` | Dựng ma trận tương tác `R` trong `load_data.py` |
| `image_list.txt` | `item_id_original item_id file_name` | Biết ảnh nào thuộc item nào để load embedding |
