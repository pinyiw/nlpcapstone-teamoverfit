from data_model import StockDataSet as SDS
import numpy as np

data = SDS('apple')

# # print(data.test_y)
# idx = 0
# batch_size = 16
# for batch_X, batch_y in data.generate_one_epoch(batch_size):
#     if idx == 0:
#         print('batch x: {}'.format(batch_X.shape))
#         print('batch y: {}'.format(batch_y.shape))
#     idx += 1
# print(idx)



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import vstack
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
        "The brown cat."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
print()

sparse_vec = sum(vector)
dense_vec = sum(vector.toarray())

dense_vec = dense_vec / np.linalg.norm(dense_vec)
print(dense_vec.shape)
sparse_vec = normalize(sparse_vec, norm='l2')
sparse_vec = vstack([sparse_vec, sparse_vec])

print(dense_vec)
print(sparse_vec.shape)
print(type(sparse_vec))
print(sparse_vec.toarray())
