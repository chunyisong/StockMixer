import random
import numpy as np
import os
import torch as torch
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss, StockMixer
import pickle
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

def save_best_model(model, loss, folder_path='models', max_keep=10):
    """
    保存模型状态字典，并保持最多max_keep个最近的模型文件。
    文件名包含时间戳和损失值（科学计数法）。
    :param model: 要保存的模型实例
    :param loss: 模型对应的损失值
    :param folder_path: 保存模型的目录路径
    :param max_keep: 最多保留的模型文件数量
    """
    # 创建目录如果不存在
    os.makedirs(folder_path, exist_ok=True)
    # 当前时间格式化为YYYYmmddHHMMSS
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    # 将浮点数损失值转换为科学记数法字符串，保留7位小数。
    formatted_loss = f"{Decimal(loss).quantize(Decimal('1e-7'), rounding=ROUND_HALF_UP):e}"
    filename = f'best_model_{now}_{formatted_loss}.pth'
    filepath = os.path.join(folder_path, filename)
    # 保存模型状态字典
    torch.save(model.state_dict(), filepath)
    # 获取目录下所有模型文件，并按修改时间排序（最新的在前）
    files = sorted([f for f in os.listdir(folder_path) if f.startswith('best_model_') and f.endswith('.pth')], reverse=True)
    # 如果超过最大保留数量，则删除多余的模型文件
    if len(files) > max_keep:
        for file_to_remove in files[max_keep:]:
            os.remove(os.path.join(folder_path, file_to_remove))

    print(f"Model checkpoint saved to {filepath} with loss {loss}.")

def load_latest_model(model, model_dir='models'):
    """
    尝试从模型目录加载最新模型到给定的模型实例。
    :param model: 要加载模型的实例
    :param model_dir: 模型存放的目录
    :return: 加载成功返回True，否则返回False
    """

    # 获取目录下所有模型文件，并按修改时间排序（最新的在前）
    files = sorted([f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pth')], reverse=True)
    if len(files):
        bestFile = files[0]
        try:
            paramsDict = torch.load(bestFile)
            model.load_state_dict(paramsDict)
            print(f"Loaded latest model checkpoint from file:{bestFile}.")
            return True
        except Exception as e:
            print(f"Failed to load the model checkpoint file:{bestFile},error:{e}")
            return False
    else:
        print("No best model checkpoint file found to load.")
        return False

np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

try_load_best_model = False
best_model_dir = '/content/stock_mixer'
data_path = './dataset'
market_name = 'NASDAQ'
relation_name = 'wikidata'
stock_num = 1026
lookback_length = 16
epochs = 100
valid_index = 756
test_index = 1008
fea_num = 5
market_num = 20
steps = 1
learning_rate = 0.001
alpha = 0.1
scale_factor = 3
activation = 'GELU'

print(f"{device},{activation},{market_name},{relation_name}")
dataset_path = data_path + '/' + market_name
if market_name == "SP500":
    data = np.load(dataset_path + '/SP500.npy')
    data = data[:, 915:, :]
    price_data = data[:, :, -1]
    mask_data = np.ones((data.shape[0], data.shape[1]))
    eod_data = data
    gt_data = np.zeros((data.shape[0], data.shape[1]))
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                   data[ticket][row - steps][-1]
else:
    with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f)

trade_dates = mask_data.shape[1]
model = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

if try_load_best_model:
    load_latest_model(model,best_model_dir)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_valid_loss = np.inf
best_test_loss = np.inf
best_valid_perf = None
best_test_perf = None
best_epoch_index = -1
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(

                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf


def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))


for epoch in range(epochs):
    print("epoch{}##########################################################".format(epoch + 1))
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    for j in range(valid_index - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(batch_offsets[j])
        )
        optimizer.zero_grad()
        prediction = model(data_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            stock_num, alpha)
        cur_loss = cur_loss
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
    print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

    if val_loss < best_valid_loss:
        best_epoch_index = epoch
        best_valid_loss = val_loss
        best_test_loss = test_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf
        save_best_model(model, best_test_loss, best_model_dir)

    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                     val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
    print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                            test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')

print('Best Valid performance:\n', 'epoch:{},loss:{:.2e}, mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(best_epoch_index,
    best_valid_loss,best_valid_perf['mse'], best_valid_perf['IC'],best_valid_perf['RIC'], best_valid_perf['prec_10'], best_valid_perf['sharpe5']))
print('Best Test performance:\n', 'epoch:{},loss:{:.2e}, mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(best_epoch_index,
    best_test_loss,best_test_perf['mse'], best_test_perf['IC'], best_test_perf['RIC'], best_test_perf['prec_10'], best_test_perf['sharpe5']))


