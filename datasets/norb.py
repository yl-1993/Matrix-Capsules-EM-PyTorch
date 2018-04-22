from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import struct


class NORB(data.Dataset):
    """`NORB <https://cs.nyu.edu/~ylclab/data/norb-v1.0/>`_ Dataset.
    The training set is composed of 5 instances of each category (instances 4, 6, 7, 8 and 9),
    and the test set of the remaining 5 instances (instances 0, 1, 2, 3, and 5). 
    The image pairs were taken by two cameras. Labels and additional info are the same
    for a stereo image pair. More details about the data collection can be found in
    `http://leon.bottou.org/publications/pdf/cvpr-2004.pdf`.
    
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    types = ['dat', 'cat', 'info']
    urls = {}

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        if len(self.urls) == 0:
            for k in self.types:
                self.urls['test_{}'.format(k)] = \
                         ['https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x01235x9x18x6x2x108x108-testing-{:02d}-{}.mat.gz' \
                         .format(x+1, k) for x in range(2)]
                self.urls['train_{}'.format(k)] = \
                         ['https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-{:02d}-{}.mat.gz'\
                         .format(x+1, k) for x in range(10)]
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # image pairs stored in [i, :, :] and [i+1, :, :]
        # they are sharing the same labels and info
        if self.train:
            self.train_data, self.train_labels, self.train_info = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            size = len(self.train_labels)
            assert size == len(self.train_info)
            assert size*2 == len(self.train_data)
            self.train_labels = self.train_labels.view(size, 1).repeat(1, 2).view(2*size, 1)
            self.train_info = self.train_info.repeat(1, 2).view(2*size, 4)
        else:
            self.test_data, self.test_labels, self.test_info = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            size = len(self.test_labels)
            assert size == len(self.test_info)
            assert size*2 == len(self.test_data)
            self.test_labels = self.test_labels.view(size, 1).repeat(1, 2).view(2*size, 1)
            self.test_info = self.test_info.repeat(1, 2).view(2*size, 4)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Note that additional info is not used in this experiment.

        Returns:
            tuple: (image, target)
            where target is index of the target class and info contains
            ...
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for k in self.urls:
            for url in self.urls[k]:
                print('Downloading ' + url)
                data = urllib.request.urlopen(url)
                filename = url.rpartition('/')[2]
                file_path = os.path.join(self.root, self.raw_folder, filename)
                with open(file_path, 'wb') as f:
                    f.write(data.read())
                with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                        gzip.GzipFile(file_path) as zip_f:
                    out_f.write(zip_f.read())
                os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        parsed = {}
        for k in self.urls:
            op = get_op(k)
            for url in self.urls[k]:
                filename = url.rpartition('/')[2].replace('.gz', '')
                path = os.path.join(self.root, self.raw_folder, filename)
                print(path)
                if k not in parsed:
                    parsed[k] = op(path)
                else:
                    parsed[k] = torch.cat([parsed[k], op(path)], dim=0)
                
        training_set = (
            parsed['train_dat'],
            parsed['train_cat'],
            parsed['train_info']
        )
        test_set = (
            parsed['test_dat'],
            parsed['test_cat'],
            parsed['test_info']
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class smallNORB(NORB):
    """`smallNORB <https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = {
        'train_dat': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz'],
        'train_cat': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz'],
        'train_info': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz'],
        'test_dat': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz'],
        'test_cat': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz'],
        'test_info': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz'],
    }


def magic2type(magic):
    m2t = {'1E3D4C51': 'single precision matrix',
           '1E3D4C52': 'packed matrix',
           '1E3D4C53': 'double precision matrix',
           '1E3D4C54': 'integer matrix',
           '1E3D4C55': 'byte matrix',
           '1E3D4C56': 'short matrix'}
    m = bytearray(reversed(magic)).hex().upper()
    return m2t[m]

def parse_header(fd):
    magic = struct.unpack('<BBBB', fd.read(4))
    ndim, = struct.unpack('<i', fd.read(4))
    dim = []
    for _ in range(ndim):
        dim += struct.unpack('<i', fd.read(4))

    header = {'magic': magic,
              'type': magic2type(magic),
              'dim': dim}
    return header

def parse_cat_file(path):
    """
        -cat file stores corresponding category of images
        Return:
            ByteTensor of shape (N,)
    """
    with open(path, 'rb') as f:
        header = parse_header(f)
        num, = header['dim']
        struct.unpack('<BBBB', f.read(4))
        struct.unpack('<BBBB', f.read(4))

        labels = np.zeros(shape=num, dtype=np.int32)
        for i in range(num):
            labels[i], = struct.unpack('<i', f.read(4))

        return torch.from_numpy(labels).long()

def parse_dat_file(path):
    """
        -dat file stores N image pairs. Each image pair, 
        [i, :, :] and [i+1, :, :], includes two images
        taken from two cameras. They share the category
        and additional information.

        Return:
            ByteTensor of shape (2*N, 96, 96)
    """
    with open(path, 'rb') as f:
        header = parse_header(f)
        num, c, h, w = header['dim']
        imgs = np.zeros(shape=(num * c, h, w), dtype=np.uint8)

        for i in range(num * c):
            img = struct.unpack('<' + h * w * 'B', f.read(h * w))
            imgs[i] = np.uint8(np.reshape(img, newshape=(h, w)))

        return torch.from_numpy(imgs)

def parse_info_file(path):
    """
        -info file stores the additional info for each image.
        The specific meaning of each dimension is:

            (:, 0): 10 instances
            (:, 1): 9 elevation
            (:, 2): 18 azimuth
            (:, 3): 6 lighting conditions

        Return:
            ByteTensor of shape (N, 4)
    """
    with open(path, 'rb') as f:
        header = parse_header(f)
        num, num_info = header['dim']
        struct.unpack('<BBBB', f.read(4))
        info = np.zeros(shape=(num, num_info), dtype=np.int32)
        for r in range(num):
            for c in range(num_info):
                info[r, c], = struct.unpack('<i', f.read(4))

        return torch.from_numpy(info)

def get_op(key):
    op_dic = {
        'train_dat': parse_dat_file,
        'train_cat': parse_cat_file,
        'train_info': parse_info_file,
        'test_dat': parse_dat_file,
        'test_cat': parse_cat_file,
        'test_info': parse_info_file
    }
    return op_dic[key]

