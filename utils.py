import numpy as np
import torch
import os
import torchvision.transforms as T
import random
from torch.utils.data import DataLoader, Dataset
import joblib

IMG_DIMS = 28


def set_random_seeds(seed=0, device='cuda:0'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Functaset(Dataset):
    def __init__(self, pkl_file):
        super(Functaset, self).__init__()
        self.functaset = joblib.load(pkl_file)

    def __getitem__(self, item):
        pair = self.functaset[item]
        modul = torch.tensor(pair['modul'])
        label = torch.tensor(pair['label'])
        return modul, label

    def __len__(self):
        return len(self.functaset)


def collate_fn(data):
    moduls, labels = zip(*data)
    return torch.stack(moduls), torch.stack(labels)

def get_fmnist_functa(
        data_dir=None,
        batch_size=256,
        mode='train',
        num_workers=2,
        pin_memory=True
):
    """
    :param data_dir: path to pickled data file
    :param batch_size: desired batch size
    :param mode: train or test 
    :param num_workers: Number of workers for data batching
    :param pin_memory:
    :return:
    """
    assert mode in ['train', 'val', 'test']
    if data_dir is None:
        data_dir = f'datasets/functaset/fmnist_{mode}.pkl'
    functaset = Functaset(data_dir)
    shuffle = mode == 'train'
    return DataLoader(
        functaset, batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_fn,
    )
    
def vec_to_img(inr, vec):
    """
    inr: a pretrained ModulatedSIREN instance - MUST PASS A SINGLE SAMPLE - this function doesn't work in batches
    vec: A modulation vector of dimension 512
    """
    return inr(vec).reshape(IMG_DIMS, IMG_DIMS)