import os
import random as rd
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from transformers import BertModel, BertTokenizer


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        # query, key, value: (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attention(query, key, value)
        return attn_output  # (seq_len, batch_size, embed_dim)

class WeightedAttention(nn.Module):
    def __init__(self, embed_dim):
        super(WeightedAttention, self).__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))  # Khởi tạo trọng số ban đầu
        self.fc = nn.Linear(embed_dim * 2, embed_dim)  # Lớp fully connected để kết hợp embeddings

    def forward(self, text_embeddings, image_embeddings):
        # Kết hợp embeddings với trọng số
        combined_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)  # Kết hợp theo chiều ngang
        # Áp dụng lớp fully connected
        combined_embeddings = self.fc(combined_embeddings)
        return combined_embeddings
    
    
class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        with open(path + '/train.txt') as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(path + '/test.txt') as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.S = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(path + '/train.txt') as f_train:
            with open(path + '/test.txt') as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        self.U = self.R.dot(self.R.transpose())
        self.I = self.R.transpose().dot(self.R)

        self.n_social = 0
        if os.path.exists(path + '/social_trust.txt'):
            with open(path + '/social_trust.txt') as f_social:
                for l in f_social.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    users = [int(i) for i in l.split(' ')]
                    uid, friends = users[0], users[1:]
                    for i in friends:
                        self.S[uid, i] = 1.
                        self.n_social = self.n_social + 1

        # Read items features data
        self.items_features = pd.read_csv(path + '/items_features.csv')
        # self.tfidf_similarity_matrix = self.create_tfidf_similarity_matrix()
        # self.bert_similarity_matrix = self.create_bert_similarity_matrix()

    def preprocess_text(self, text):
        text = text.lower()
        return text


    def get_bert_embeddings_batch(self, texts, batch_size=32):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            with torch.no_grad():
                outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings).astype(np.float32)
    
    
    def create_tfidf_similarity_matrix(self):
        self.items_features['combined_features'] = self.items_features['feature1'] + ' ' + self.items_features[
            'feature2']
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.items_features['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Set values below 0.5 to 0
        cosine_sim[cosine_sim < 0.5] = 0

        similarity_matrix = sp.csr_matrix(cosine_sim)
        return similarity_matrix

    def create_bert_similarity_matrix(self):
        self.items_features['cleaned_feature2'] = self.items_features['feature2'].apply(self.preprocess_text)

        # Load BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Function to get BERT embeddings
        def get_bert_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings

        # Get BERT embeddings for feature2
        bert_embeddings = np.vstack(
            self.items_features['cleaned_feature2'].apply(lambda x: get_bert_embeddings(x)).to_numpy()
        )

        bert_similarity = cosine_similarity(bert_embeddings, bert_embeddings)

        # Set values below 0.5 to 0
        bert_similarity[bert_similarity < 0.5] = 0

        bert_similarity_matrix = sp.csr_matrix(bert_similarity)

        # Get TF-IDF vectors for feature1
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.items_features['feature1'])
        tfidf_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Set values below 0.5 to 0
        tfidf_similarity[tfidf_similarity < 0.5] = 0

        tfidf_similarity_matrix = sp.csr_matrix(tfidf_similarity)

        # Combine the two similarity matrices by multiplication
        combined_similarity_matrix = bert_similarity_matrix.multiply(tfidf_similarity_matrix)

        return combined_similarity_matrix

    def create_full_bert_similarity_matrix(self):

        self.items_features['combined_features'] = self.items_features['feature1'] + ' ' + self.items_features['feature2']

        self.items_features['cleaned_combined_features'] = self.items_features['combined_features'].apply(self.preprocess_text)

        # Load BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Function to get BERT embeddings
        def get_bert_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings

        # Get BERT embeddings for feature2
        bert_embeddings = np.vstack(
            self.items_features['cleaned_combined_features'].apply(lambda x: get_bert_embeddings(x)).to_numpy()
        )

        bert_similarity = cosine_similarity(bert_embeddings, bert_embeddings)

        # Set values below 0.5 to 0
        bert_similarity[bert_similarity < 0.5] = 0

        bert_similarity_matrix = sp.csr_matrix(bert_similarity)

        return bert_similarity_matrix



#TODO: START ###################################################################################################



    def create_img_similarity_matrix(self, threshold=0.5):
        """
        Tạo ma trận thưa từ các embeddings hình ảnh trong self.items_features['feature3'].
        
        Args:
            threshold: Ngưỡng cắt để tạo ma trận thưa (mặc định là 0.5).
        
        Returns:
            sparse_similarity_matrix: Ma trận thưa dạng sparse matrix.
        """
        
        # Kiểm tra xem cột 'feature3' có tồn tại không
        if 'feature3' not in self.items_features.columns:
            raise ValueError("Cột 'feature3' không tồn tại trong items_features.")
        
        # 1. Chuyển đổi các embeddings từ string sang numpy array
        def parse_vector_string(vector_string):
            vector = vector_string.strip('[]').split()
            return np.array([float(x) for x in vector], dtype=np.float32)
        
        image_embeddings = np.vstack(
            self.items_features['feature3'].apply(parse_vector_string).values
        ).astype(np.float32)
        
        print("Image embeddings shape:", image_embeddings.shape)  # Should be (n_samples, 768)
        
        # 2. Tính toán ma trận tương đồng cosine
        similarity_matrix = cosine_similarity(image_embeddings, image_embeddings)
        
        # 3. Áp dụng ngưỡng cắt
        similarity_matrix[similarity_matrix < threshold] = 0
        
        # 4. Chuyển đổi sang ma trận thưa
        sparse_similarity_matrix = sp.csr_matrix(similarity_matrix)
        
        return sparse_similarity_matrix


    def create_multimodal_similarity_matrix(self, method="late_fusion", alpha=0.5, pca_components=256):
        """
        Tạo similarity matrix bằng cách kết hợp text_embeddings và image_embeddings.
        
        Args:
            method: Phương pháp kết hợp đặc trưng. Bao gồm:
                - "late_fusion" (default): Tính ma trận tương đồng riêng biệt và kết hợp bằng trọng số.
                - "aggregation": Kết hợp đặc trưng bằng trung bình hoặc tổng.
                - "pca": Giảm chiều embeddings trước khi kết hợp.
                - "attention": Sử dụng cơ chế attention để học trọng số tự động.
            alpha: Trọng số cho phương pháp late fusion (mặc định là 0.5).
            pca_components: Số chiều sau giảm chiều bằng PCA (nếu sử dụng phương pháp PCA).
        
        Returns:
            similarity_matrix: Ma trận tương đồng dạng sparse matrix.
        """
        
        # Kiểm tra các cột dữ liệu cần thiết
        if 'feature1' not in self.items_features.columns or 'feature2' not in self.items_features.columns or 'feature3' not in self.items_features.columns:
            raise ValueError("Các cột 'feature1', 'feature2', và 'feature3' phải tồn tại trong items_features.")

        # 1. Xử lý đặc trưng văn bản
        self.items_features['combined_text'] = self.items_features['feature1'] + ' ' + self.items_features['feature2']
        self.items_features['cleaned_combined_text'] = self.items_features['combined_text'].apply(self.preprocess_text)
        
        # Tạo BERT embeddings cho văn bản
        text_embeddings = self.get_bert_embeddings_batch(self.items_features['cleaned_combined_text'].tolist())
        print("Text embeddings shape:", text_embeddings.shape)  # Should be (n_samples, 768)
        
        # Chuyển string vector thành numpy array và chuyển về float32
        def parse_vector_string(vector_string):
            vector = vector_string.strip('[]').split()
            return np.array([float(x) for x in vector], dtype=np.float32)

        image_embeddings = np.vstack(
            self.items_features['feature3'].apply(parse_vector_string).values
        ).astype(np.float32)
        print("Image embeddings shape:", image_embeddings.shape)  # Should be (n_samples, 768)
        
        # 2. Kết hợp đặc trưng theo phương pháp được chọn
        if method == "late_fusion":
            # Tính ma trận tương đồng riêng biệt
            text_similarity = cosine_similarity(text_embeddings, text_embeddings)
            image_similarity = cosine_similarity(image_embeddings, image_embeddings)

            # Kết hợp ma trận tương đồng với trọng số alpha
            combined_similarity = alpha * text_similarity + (1 - alpha) * image_similarity

        elif method == "aggregation":
            # Kết hợp embeddings bằng trung bình
            combined_embeddings = (text_embeddings + image_embeddings) / 2

            # Tính ma trận tương đồng
            combined_similarity = cosine_similarity(combined_embeddings, combined_embeddings)

        elif method == "pca":
            if pca_components is None:
                raise ValueError("Bạn cần chỉ định số chiều PCA thông qua tham số 'pca_components'.")

            # Kết hợp embeddings trước khi giảm chiều
            combined_embeddings = np.concatenate([text_embeddings, image_embeddings], axis=1)

            # Giảm chiều embeddings bằng PCA
            pca = PCA(n_components=pca_components)
            reduced_embeddings = pca.fit_transform(combined_embeddings)

            # Tính ma trận tương đồng
            combined_similarity = cosine_similarity(reduced_embeddings, reduced_embeddings)

        elif method == "attention":
            # Chuyển embeddings sang tensor
            text_tensor = torch.tensor(text_embeddings, dtype=torch.float32)
            image_tensor = torch.tensor(image_embeddings, dtype=torch.float32)

            # Sử dụng Multi-Head Attention
            multihead_attention = MultiHeadAttention(embed_dim=768, num_heads=8)
            text_attended = multihead_attention(text_tensor, text_tensor, text_tensor)
            image_attended = multihead_attention(image_tensor, image_tensor, image_tensor)

            # Sử dụng Weighted Attention để kết hợp embeddings
            weighted_attention = WeightedAttention(embed_dim=768)
            combined_embeddings = weighted_attention(text_attended, image_attended)

            # Tính ma trận tương đồng
            combined_similarity = cosine_similarity(combined_embeddings.detach().numpy(), combined_embeddings.detach().numpy())

        else:
            raise ValueError("Phương pháp kết hợp không hợp lệ. Chọn 'late_fusion', 'aggregation', 'pca', hoặc 'attention'.")

        # 3. Áp dụng ngưỡng cắt (threshold)
        combined_similarity[combined_similarity < 0.5] = 0

        # 4. Chuyển đổi sang sparse matrix
        return sp.csr_matrix(combined_similarity)
    
#TODO: END ###################################################################################################
    
    def get_norm_adj_mat(self):
        def normalized_sym(adj):
            rowsum: ndarray = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocsr()

        try:
            t1 = time()
            interaction_adj_mat_sym = sp.load_npz(self.path + '/s_interaction_adj_mat.npz')
            print('already load interaction adj matrix', interaction_adj_mat_sym.shape, time() - t1)
        except Exception:
            interaction_adj_mat = self.create_interaction_adj_mat()
            interaction_adj_mat_sym = normalized_sym(interaction_adj_mat)
            print('generate symmetrically normalized interaction adjacency matrix.')
            sp.save_npz(self.path + '/s_interaction_adj_mat.npz', interaction_adj_mat_sym)

        try:
            t2 = time()
            social_adj_mat_sym = sp.load_npz(self.path + '/s_social_adj_mat.npz')
            print('already load social adj matrix', social_adj_mat_sym.shape, time() - t2)
        except Exception:
            social_adj_mat = self.create_social_adj_mat()
            social_adj_mat_sym = normalized_sym(social_adj_mat)
            print('generate symmetrically normalized social adjacency matrix.')
            sp.save_npz(self.path + '/s_social_adj_mat.npz', social_adj_mat_sym)
        
        try:
            t3 = time()
            similar_users_adj_mat_sym = sp.load_npz(self.path + '/s_similar_users_adj_mat.npz')
            print('already load similar users adj matrix', similar_users_adj_mat_sym.shape, time() - t3)
        except Exception:
            similar_users_adj_mat = self.create_similar_adj_mat()
            similar_users_adj_mat_sym = normalized_sym(similar_users_adj_mat)
            print('generate symmetrically normalized similar users adjacency matrix.')
            sp.save_npz(self.path + '/s_similar_users_adj_mat.npz', similar_users_adj_mat_sym)


        try:
            t4 = time()
            tfidf_item_similarity_adj_mat_sym = sp.load_npz(self.path + '/s_tfidf_item_similarity_adj_mat.npz')
            print('already load tfidf item similarity adj matrix', tfidf_item_similarity_adj_mat_sym.shape, time() - t4)
        except Exception:
            tfidf_item_similarity_adj_mat = self.create_tfidf_similarity_matrix()
            tfidf_item_similarity_adj_mat_sym = normalized_sym(tfidf_item_similarity_adj_mat)
            print('generate symmetrically normalized tfidf item similarity adjacency matrix.')
            sp.save_npz(self.path + '/s_tfidf_item_similarity_adj_mat.npz', tfidf_item_similarity_adj_mat_sym)


        try:
            t5 = time()
            bert_item_similarity_adj_mat_sym = sp.load_npz(self.path + '/s_bert_item_similarity_adj_mat.npz')
            print('already load bert item similarity adj matrix', bert_item_similarity_adj_mat_sym.shape, time() - t5)
        except Exception:
            bert_item_similarity_adj_mat = self.create_bert_similarity_matrix()
            bert_item_similarity_adj_mat_sym = normalized_sym(bert_item_similarity_adj_mat)
            print('generate symmetrically normalized bert item similarity adjacency matrix.')
            sp.save_npz(self.path + '/s_bert_item_similarity_adj_mat.npz', bert_item_similarity_adj_mat_sym)

        try:
            t6 = time()
            full_bert_item_similarity_adj_mat_sym = sp.load_npz(self.path + '/s_full_bert_item_similarity_adj_mat.npz')
            print('already load full bert item similarity adj matrix', full_bert_item_similarity_adj_mat_sym.shape,
                  time() - t6)
        except Exception:
            full_bert_item_similarity_adj_mat = self.create_full_bert_similarity_matrix()
            full_bert_item_similarity_adj_mat_sym = normalized_sym(full_bert_item_similarity_adj_mat)
            print('generate symmetrically normalized full bert item similarity adjacency matrix.')
            sp.save_npz(self.path + '/s_full_bert_item_similarity_adj_mat.npz', full_bert_item_similarity_adj_mat_sym)




#TODO: START ###################################################################################################

        # # Thêm phần xử lý multimodal similarity matrix
        # try:
        #     t7 = time()
        #     multimodal_similarity_adj_mat_sym = sp.load_npz(self.path + '/s_multimodal_similarity_adj_mat.npz')
        #     print('already load multimodal similarity adj matrix', multimodal_similarity_adj_mat_sym.shape, time() - t7)
        # except Exception:
        #     multimodal_similarity_adj_mat = self.create_multimodal_similarity_matrix(method="attention", alpha=0.5, pca_components=768)
        #     multimodal_similarity_adj_mat_sym = normalized_sym(multimodal_similarity_adj_mat)
        #     print('generate symmetrically normalized multimodal similarity adjacency matrix.')
        #     sp.save_npz(self.path + '/s_multimodal_similarity_adj_mat.npz', multimodal_similarity_adj_mat_sym)
        
        
        # # Thêm phần xử lý img similarity matrix
        # try:
        #     t8 = time()
        #     img_similarity_adj_mat_sym = sp.load_npz(self.path + '/s_img_similarity_adj_mat.npz')
        #     print('already load img similarity adj matrix', img_similarity_adj_mat_sym.shape, time() - t8)
        # except Exception:
        #     img_similarity_adj_mat = self.create_img_similarity_matrix(threshold=0.5)
        #     img_similarity_adj_mat_sym = normalized_sym(img_similarity_adj_mat)
        #     print('generate symmetrically normalized img similarity adjacency matrix.')
        #     sp.save_npz(self.path + '/s_img_similarity_adj_mat.npz', img_similarity_adj_mat_sym)
        # The code snippet is a Python function that seems to be returning multiple adjacency matrices related to interactions, social connections, user similarities, item similarities, and image similarities. The comment indicates that a new adjacency matrix called `multimodal_similarity_adj_mat_sym` is being added to the return statement.
        # Thêm multimodal_similarity_adj_mat_sym vào return
        # return interaction_adj_mat_sym, social_adj_mat_sym, similar_users_adj_mat_sym, \
        #     tfidf_item_similarity_adj_mat_sym, bert_item_similarity_adj_mat_sym, \
        #     multimodal_similarity_adj_mat_sym, \
        #     img_similarity_adj_mat_sym
            
        return interaction_adj_mat_sym, social_adj_mat_sym, similar_users_adj_mat_sym, \
            tfidf_item_similarity_adj_mat_sym, bert_item_similarity_adj_mat_sym, \
            full_bert_item_similarity_adj_mat_sym
        
        # return interaction_adj_mat_sym, social_adj_mat_sym, similar_users_adj_mat_sym, tfidf_item_similarity_adj_mat_sym, tfidf_item_similarity_adj_mat_sym


#TODO: START ###################################################################################################

    def create_interaction_adj_mat(self):
        # 1. Create Graph Users-Items  Interaction Adjacency Matrix.
        t1 = time()
        interaction_adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
                                            dtype=np.float32)
        interaction_adj_mat = interaction_adj_mat.tolil()
        R = self.R.tolil()
        for i in range(5):
            interaction_adj_mat[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5), self.n_users:] = \
                R[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)]
            interaction_adj_mat[self.n_users:, int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)] = \
                R[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)].T
        print('already create interaction adjacency matrix', interaction_adj_mat.shape, time() - t1)
        return interaction_adj_mat.tocsr()

    def create_social_adj_mat(self):
        # 2. Create Graph Users-Users Social Adjacency Matrix.
        t2 = time()
        social_adj_mat = self.S
        print('already create social adjacency matrix', social_adj_mat.shape, ' - social_interactons:', self.n_social,
              time() - t2)
        return social_adj_mat.tocsr()

    def create_similar_adj_mat(self):
        t3 = time()
        similar_users_adj_mat = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)

        def cluster(x, d, t):
            v = x / (t + d - x)
            if v <= 0.09:
                return 0
            elif 0.09 < v <= 0.39:
                return 1
            elif 0.49 < v <= 0.69:
                return 10
            elif 0.69 < v <= 0.79:
                return 100
            else:
                return 200

        X = self.U.toarray()
        vfunc = np.vectorize(cluster)

        diag: ndarray = X.diagonal()
        for i in range(X.shape[0]):
            tmp = vfunc(X[i], diag, diag[i])
            similar_users_adj_mat[i] = tmp
        print('already create similar users adjacency matrix', similar_users_adj_mat.shape, time() - t3)
        similar_users_adj_mat = sp.csr_matrix(similar_users_adj_mat)

        return similar_users_adj_mat.tocsr()

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u] + self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))
