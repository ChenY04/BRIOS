import torch
import torch.optim as optim
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
import models
from support.early_stopping import EarlyStopping
import batch_data_loader
from math import sqrt
from sklearn import metrics
from tqdm import tqdm

# parameter setting
batch_size = 512
model_name = 'brios'
hid_size = 96
SEQ_LEN = 46
INPUT_SIZE = 3
SELECT_SIZE = 1

# predicting process
def ExecuteModel():

    print('=======')
    print('predicting')
    print('=======')
    model = getattr(models, model_name).Model(hid_size, INPUT_SIZE, SEQ_LEN, SELECT_SIZE)
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    SavePath = '**/**.pt'  #Model parameter path

    model.load_state_dict(torch.load(SavePath))

    # load input data
    data_path = '**/**.json'
    data_iter = batch_data_loader.get_test_loader(batch_size=batch_size, prepath=data_path)

    model.eval()

    evals = []
    imputations = []

    save_impute = []

    for idx, data in tqdm(enumerate(data_iter)):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        masks = ret['masks'].data.cpu().numpy()

        count_ones = np.count_nonzero(eval_masks == 1)

        if count_ones != 0:
            evals += eval_[np.where(eval_masks == 1)].tolist()
            imputations += imputation[np.where(eval_masks == 1)].tolist()

        count_ones = np.count_nonzero(masks == 0)

        if count_ones != 0:
            imputation_fill = imputation[np.where(masks == 0)]
            save_impute.append(imputation_fill)
        del eval_, eval_masks, imputation, imputation_fill, masks

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    if evals.size != 0:
        print('MAE', np.abs(evals - imputations).mean())
        print('RMSE', sqrt(metrics.mean_squared_error(evals, imputations)))

    del evals, imputations

    save_impute_p = np.asarray(save_impute, dtype=object)
    if save_impute_p.size != 0:
        save_impute = np.concatenate(save_impute, axis=0)
        save_impute = (save_impute * 10000).astype(np.int16)
    else:
        save_impute = np.asarray(save_impute, dtype=object)

    resultpath = '**/**'   #predicted values save path
    np.save(resultpath, save_impute)

    del save_impute, data_iter, data, ret, model


if __name__ == '__main__':
    ExecuteModel()
