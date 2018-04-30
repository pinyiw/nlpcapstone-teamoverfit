from data_model import StockDataSet as SDS

data = SDS('apple')

# print(data.test_y)
idx = 0
batch_size = 16
for batch_X, batch_y in data.generate_one_epoch(batch_size):
    if idx == 0:
        print('batch x: {}'.format(batch_X.shape))
        print('batch y: {}'.format(batch_y.shape))
    idx += 1
print(idx)