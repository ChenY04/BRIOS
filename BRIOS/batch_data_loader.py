import os
import ujson as json
import numpy as np
from scipy.signal import savgol_coeffs
from scipy.ndimage import convolve1d
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class MyTrainSet(Dataset):
    def __init__(self, prepath):
        super(MyTrainSet, self).__init__()
        self.prepath = prepath
        self.content = open(self.prepath).readlines()

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        # print(idx)
        rec = json.loads(self.content[idx])

        rec['is_train'] = 1
        return rec

class MyTestSet(Dataset):
    def __init__(self, prepath):
        super(MyTestSet, self).__init__()
        self.prepath = prepath
        self.content = open(self.prepath).readlines()

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])

        rec['is_train'] = 0
        return rec


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))
    # normalize SAR data
    scaler = MinMaxScaler(feature_range=[-1, 1])

    def to_tensor_dict(recs):

        # extract SAR temporal change trend
        coeffs_long_trend = savgol_coeffs(19, 2)
        values0 = np.array(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        sar1 = values0[:, :, 0]
        sar2 = values0[:, :, 1]
        bsize = values0.shape[0]
        for k in range(bsize):
            onesar1 = sar1[k:k+1, :].T
            onesar2 = sar2[k:k+1, :].T
            onesar1 = scaler.fit_transform(onesar1)
            onesar2 = scaler.fit_transform(onesar2)
            onesar1 = onesar1[:, 0]
            onesar2 = onesar2[:, 0]
            onesar10 = convolve1d(onesar1, coeffs_long_trend, mode="wrap")
            onesar20 = convolve1d(onesar2, coeffs_long_trend, mode="wrap")
            values0[k, :, 0] = onesar10
            values0[k, :, 1] = onesar20
        values = torch.FloatTensor(values0)

        # values = torch.FloatTensor(
        #     list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))

        masks = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
        deltas = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        eval_masks = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))

        return {'values': values, 'masks': masks, 'deltas': deltas,'eval_masks': eval_masks}

    ret_dict = {
        'forward': to_tensor_dict(forward),
        'backward': to_tensor_dict(backward)
    }

    ret_dict['is_train'] = torch.FloatTensor(
        list(map(lambda x: x['is_train'], recs)))

    return ret_dict

def get_train_loader(batch_size, prepath, shuffle=True):
    data_set = MyTrainSet(prepath)
    data_iter = DataLoader(dataset=data_set, \
                           batch_size=batch_size, \
                           num_workers=8, \
                           shuffle=shuffle, \
                           pin_memory=True, \
                           collate_fn=collate_fn
                           )
    return data_iter

def get_test_loader(batch_size, prepath, shuffle=False):
    data_set = MyTestSet(prepath)
    data_iter = DataLoader(dataset=data_set, \
                           batch_size=batch_size, \
                           num_workers=8, \
                           shuffle=shuffle, \
                           pin_memory=True, \
                           collate_fn=collate_fn
                           )

    return data_iter


