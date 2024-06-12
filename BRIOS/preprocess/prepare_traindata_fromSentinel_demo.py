import numpy as np
import os
from osgeo import gdal
import pandas as pd
import ujson as json
from tqdm import tqdm

def import_data(path):
    raster_dataset = gdal.Open(path, gdal.GA_ReadOnly)
    bands_data = []
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    bands_data = np.dstack(bands_data)
    return bands_data

def write_geotiff(fname, data, geo_transform, projection):
    """Create a GeoTIFF file with the given data."""
    driver = gdal.GetDriverByName('ENVI')
    rows, cols, nbands = data.shape[:3]
    # rows, cols = data.shape[:3]
    # nbands = 1
    dataset = driver.Create(fname, cols, rows, nbands, gdal.GDT_Float16)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)

    if nbands == 1:
        dataset.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(nbands):
            dataset.GetRasterBand(i + 1).WriteArray(data[:, :, i])
    del dataset

def get_consecutive_count(origin):
    origin = origin.tolist()
    g = [origin[:1]]
    [g[-1].append(y) if x == y else g.append([y]) for x, y in zip(origin[:-1], origin[1:])]
    output = np.zeros(len(origin))
    start = 0
    for i in range(len(g)):
        oneg = np.array(g[i])
        leng = len(oneg)
        end = start+leng
        if np.mean(oneg) != 0:
            output[start:end] = leng
        else:
            output[start:end] = 0
        start = end

    return output

def cal_timestep(time, mask):
    deltaT = time.copy()
    for i in range(len(time)):
        T_time0 = time[i]
        if i != 0:
            for k in range(i - 1, -1, -1):
                T_time1 = time[k]
                if mask[k] == 1:
                    T_time1 = time[k]
                    break

            T = T_time0-T_time1
        else:
            T = 0

        deltaT[i] = T

    return deltaT

def savePreprocessedData(path, data):
    with open(path +".npy", 'bw') as outfile:
        np.save(outfile, data)

def parse_rec(values, masks, eval_masks, deltas):

    rec = []
    for i in range(values.shape[0]):
        recone = {}
        recone['deltas'] = deltas[i, :].tolist()
        recone['masks'] = masks[i].astype('int8').tolist()
        recone['values'] = values[i, :].tolist()
        recone['eval_masks'] = eval_masks[i].astype('int8').tolist()
        rec.append(recone)

    return rec

def parse_idTrain(id_):
    values = traindatasets_valuesF[:, :, id_]
    masks = traindatasets_maskF[:, id_]
    eval_masks = traindatasets_evalmaskF[:, id_]
    deltas = traindatasets_deltaF[:, :, id_]
    deltasB = traindatasets_deltaBF[:, :, id_]

    rec = {}

    rec['forward'] = parse_rec(values, masks, eval_masks, deltas)
    rec['backward'] = parse_rec(values[::-1], masks[::-1], eval_masks[::-1], deltasB)

    rec = json.dumps(rec)

    fs.write(rec + '\n')

print('*************')
print('from image series data to training datasets：')
areamask_path = r"**\**"  #study area mask: 0-invalid/no data; 1-valid
cloudmask_path = r"**\**"  #cloud mask: 0-clear observations; 1-real cloud; 2-simulated cloud

areamask = import_data(areamask_path)
cloudmask = import_data(cloudmask_path)
datanum = np.apply_along_axis(lambda x: np.count_nonzero(np.logical_not(x)), axis=2, arr=cloudmask)

rows, cols, n_bands = cloudmask.shape

n_samples = rows * cols
areamask0 = areamask.reshape((n_samples))
datanum0 = datanum.reshape((n_samples))
del areamask


# training data splitting
idx = np.argwhere((areamask0 != 0) & (datanum0 > 25))
train_index = np.random.choice(idx.flatten(), size=800000, replace=False)
savePreprocessedData("**\**",train_index)

# use prepared training data splitting file
# train_index = np.load("**\**")
# trainmask = np.zeros(n_samples)
# trainmask[train_index] = 1
# trainmask = trainmask.reshape((rows, cols))

# processing large areas in blocks, Partition the dataset in advance by spatial blocks
batch_flag = ['1','2','3','4']
x1 = [0,0,5492,5492]
x2 = [5492,5492,cols,cols]
y1 = [0,5491,0,5491]
y2 = [5491,rows,5491,rows]

# number of features and time step of your interest
feature_num = 3
time = np.arange(1,369,8)

traindatasets_valuesF = np.empty((n_bands, 3, 0),dtype=np.float16)
traindatasets_evalmaskF = np.empty((n_bands, 0),dtype=np.int8)
traindatasets_maskF = np.empty((n_bands, 0),dtype=np.int8)
traindatasets_deltaF = np.empty((n_bands, 3, 0),dtype=np.float16)
traindatasets_deltaBF = np.empty((n_bands, 3, 0),dtype=np.float16)

for kk in range(len(batch_flag)):
    ndvi_data_path = r"**\**"+batch_flag[kk]
    vh_data_path = r"**\**"+batch_flag[kk]
    rvi_data_path = r"**\**"+batch_flag[kk]

    ndvi_data = import_data(ndvi_data_path)
    vh_data = import_data(vh_data_path)
    rvi_data = import_data(rvi_data_path)

    rows0, cols0, n_bands0 = ndvi_data.shape

    cloudmask0 = cloudmask[y1[kk]:y2[kk],x1[kk]:x2[kk],:]
    trainmask0 = trainmask[y1[kk]:y2[kk],x1[kk]:x2[kk]]

    n_samples0 = rows0 * cols0
    ndvi_data0 = ndvi_data.reshape((n_samples0, n_bands0))
    vh_data0 = vh_data.reshape((n_samples0, n_bands0))
    rvi_data0 = rvi_data.reshape((n_samples0, n_bands0))
    cloudmask_arr = cloudmask0.reshape((n_samples0, n_bands0))
    trainmask_arr = trainmask0.reshape((n_samples0))
    train_index0 = np.where(trainmask_arr == 1)
    train_index0 = train_index0[0]

    del ndvi_data,vh_data,rvi_data,cloudmask0,trainmask0

    ndvi_dataT = ndvi_data0[train_index0, :]
    vh_dataT = vh_data0[train_index0, :]
    rvi_dataT = rvi_data0[train_index0, :]
    cloudmask_arrT = cloudmask_arr[train_index0, :]
    maskT =  np.where(cloudmask_arrT == 0, 1, 0)
    eval_maskT = np.where(cloudmask_arrT == 2, 1, 0)

    del ndvi_data0,vh_data0,rvi_data0,cloudmask_arr

    print('Generate time interval for training dataset: ')
    deltaT = np.zeros((len(train_index0), n_bands0))
    for i in tqdm(range(len(train_index0))):
        maskone = maskT[i, :]
        done = cal_timestep(time, maskone)
        deltaT[i, :] = done

    print('Generate backward time interval for training dataset: ')
    deltaTb = np.zeros((len(train_index0), n_bands0))
    for i in tqdm(range(len(train_index0))):
        maskone = maskT[i, :]
        maskone = maskone[::-1]
        done = cal_timestep(time, maskone)
        deltaTb[i, :] = done

    print('Generate time interval for SAR data: ')
    deltaTt = np.zeros((len(train_index0), n_bands0))
    for i in tqdm(range(len(train_index0))):
        maskone = np.ones(n_bands0)
        maskone = np.int_(maskone)
        done = cal_timestep(time, maskone)
        deltaTt[i, :] = done

    print('Generate training dataset: ')
    traindatasets_values = np.zeros((n_bands0, feature_num, len(train_index0)),dtype=np.float16)
    for i in tqdm(range(n_bands0)):
        for k in range(len(train_index0)):
            traindatasets_values[i, 0, k] = vh_dataT[k, i]
            traindatasets_values[i, 1, k] = rvi_dataT[k, i]
            traindatasets_values[i, 2, k] = ndvi_dataT[k, i]
    traindatasets_valuesF = np.concatenate((traindatasets_valuesF, traindatasets_values), axis=2)
    del vh_dataT,rvi_dataT,ndvi_dataT

    print('Generate evalmask: ')  #evalmask: where is the validation/simulated data used as evaluation
    traindatasets_evalmask = np.zeros((n_bands0, len(train_index0)),dtype=np.int8)
    for i in tqdm(range(n_bands0)):
        for k in range(len(train_index0)):
            traindatasets_evalmask[i, k] = eval_maskT[k, i]
    traindatasets_evalmaskF = np.concatenate((traindatasets_evalmaskF, traindatasets_evalmask), axis=1)

    print('Generate mask: ')   #mask: where is cloudy/missing data including real and simulated one
    traindatasets_mask = np.zeros((n_bands0, len(train_index0)),dtype=np.int8)
    for i in tqdm(range(n_bands0)):
        for k in range(len(train_index0)):
            traindatasets_mask[i, k] = maskT[k, i]
    traindatasets_maskF = np.concatenate((traindatasets_maskF, traindatasets_mask), axis=1)
    del eval_maskT,maskT

    print('Generate training dataset: ')
    traindatasets_delta = np.zeros((n_bands0, feature_num, len(train_index0)),dtype=np.float16)
    for i in tqdm(range(n_bands0)):
        for k in range(len(train_index0)):
            traindatasets_delta[i, 0, k] = deltaTt[k, i]
            traindatasets_delta[i, 1, k] = deltaTt[k, i]
            traindatasets_delta[i, 2, k] = deltaT[k, i]
    traindatasets_deltaF = np.concatenate((traindatasets_deltaF, traindatasets_delta), axis=2)

    print('Generate training dataset (backward): ')
    traindatasets_deltaB = np.zeros((n_bands0, feature_num, len(train_index0)),dtype=np.float16)
    for i in tqdm(range(n_bands0)):
        for k in range(len(train_index0)):
            traindatasets_deltaB[i, 0, k] = deltaTt[k, i]
            traindatasets_deltaB[i, 1, k] = deltaTt[k, i]
            traindatasets_deltaB[i, 2, k] = deltaTb[k, i]
    traindatasets_deltaBF = np.concatenate((traindatasets_deltaBF, traindatasets_deltaB), axis=2)

    del deltaTt,deltaTb,traindatasets_values, traindatasets_evalmask, traindatasets_mask, traindatasets_delta, traindatasets_deltaB

all_len = traindatasets_valuesF.shape[2]
fs = open('**/**.json', 'w')
print('save training dataset as json: ')
for id_ in tqdm(range(all_len)):
    parse_idTrain(id_)

del traindatasets_valuesF,traindatasets_evalmaskF,traindatasets_maskF,traindatasets_deltaF,traindatasets_deltaBF
fs.close()
