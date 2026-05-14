"""
batch_test.py — Evaluation cho CombiGCN PyG
=============================================
Convert từ hr/utility/batch_test.py (TF Session → PyTorch).
"""

import numpy as np
import torch
from evaluator import eval_score_matrix_foldout


def test(model, interaction_adj, similarity_adj, data, Ks, device,
         batch_size=1024, train_set_flag=0):
    """
    Evaluate model trên test set (hoặc train set).

    Args:
        model:           CombiGCN model
        interaction_adj: SparseTensor bipartite adj
        similarity_adj:  SparseTensor item similarity (hoặc None cho LightGCN thuần)
        data:            Data object (có .test_set, .train_items)
        Ks:              list of K values, e.g. [1, 5, 10, 20, 50]
        device:          torch device
        batch_size:      user batch size cho evaluation
        train_set_flag:  0 = evaluate trên test set, 1 = evaluate trên train set

    Returns:
        dict: {'precision': array, 'recall': array, 'ndcg': array,
               'map': array, 'hit_ratio': array, 'mrr': array}
    """
    model.eval()

    top_show = np.sort(Ks)
    max_top = max(top_show)

    if train_set_flag == 0:
        users_to_test = list(data.test_set.keys())
    else:
        users_to_test = list(data.train_items.keys())

    n_test_users = len(users_to_test)
    all_result = []

    for start in range(0, n_test_users, batch_size):
        end = min(start + batch_size, n_test_users)
        user_batch = users_to_test[start:end]

        # ── Predict scores (PyTorch) ──
        user_tensor = torch.LongTensor(user_batch).to(device)
        rate_batch = model.predict(interaction_adj, similarity_adj, user_tensor)
        rate_batch = rate_batch.cpu().numpy()  # (batch, n_items)

        # ── Build test items + mask training items ──
        test_items = []
        if train_set_flag == 0:
            for user in user_batch:
                test_items.append(data.test_set[user])
            # Mask training items → -inf
            for idx, user in enumerate(user_batch):
                train_items_off = data.train_items[user]
                rate_batch[idx][train_items_off] = -np.inf
        else:
            for user in user_batch:
                test_items.append(data.train_items[user])

        # ── Evaluate batch ──
        batch_result = eval_score_matrix_foldout(rate_batch, test_items, max_top)
        all_result.append(batch_result)

    all_result = np.concatenate(all_result, axis=0)
    assert len(all_result) == n_test_users

    final_result = np.mean(all_result, axis=0)
    final_result = np.reshape(final_result, newshape=[6, max_top])
    final_result = final_result[:, top_show - 1]
    final_result = np.reshape(final_result, newshape=[6, len(top_show)])

    result = {
        "precision":  final_result[0],
        "recall":     final_result[1],
        "map":        final_result[2],
        "ndcg":       final_result[3],
        "mrr":        final_result[4],
        "hit_ratio":  final_result[5],
    }
    return result
