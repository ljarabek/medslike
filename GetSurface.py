from tqdm import tqdm
import CT
import numpy as np
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')
cfgt = config['TUMOR BOUNDARIES']
cfgr = config['REGION BOUNDARIES']
cfg = config['DEFAULT']
BatchSize = int(cfg['batchSize'])
TrainMaxIndex = int(cfg['trainMaxIndex'])
testFrom = int(cfg['testFrom'])
testTo = int(cfg['testTo'])
regularization = float(cfg['regularization'])
path = cfg['MHA_path']



if not os.path.exists(path + 'Surface/'):
    os.makedirs(path + 'Surface/')

for i in tqdm(range(testTo)):
    example = CT.SurfaceAsCoordinates(i)
    #example = CT.crop2DArr(example,17,211,118,397)
    np.save(path + 'surface/{}.npy'.format(i), example)


vse = []
for i in range(testTo):
    vse.append(np.load(path + 'surface/{}.npy'.format(i)))
print(np.array(vse).shape)


np.save(path + 'surface/all.npy', vse)