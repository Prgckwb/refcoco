"""Refer to https://github.com/open-mmlab/mmdetection/blob/main/tools/misc/download_dataset.py"""
import argparse
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download datasets for training')
    parser.add_argument(
        '--dataset-name',
        type=str,
        help='dataset name',
        default='refcoco',
        choices=['coco2014', 'refcoco']
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default='data/coco')
    parser.add_argument(
        '--unzip',
        action='store_true',
        help='whether unzip dataset or not, zipped files will be saved')
    parser.add_argument(
        '--delete',
        action='store_true',
        help='delete the download zipped files')
    parser.add_argument(
        '--threads', type=int, help='number of threading', default=4)
    args = parser.parse_args()
    return args


def download(url, dir, unzip=True, delete=False, threads=1):
    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}')
            torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.tar'):
            print(f'Unzipping {f.name}')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.tar':
                TarFile(f).extractall(path=dir)
            if delete:
                f.unlink()
                print(f'Delete {f}')

    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


if __name__ == '__main__':
    args = parse_args()
    path = Path(args.save_dir)

    data2url = dict(
        coco2014=[
            'http://images.cocodataset.org/zips/train2014.zip',
            'http://images.cocodataset.org/zips/val2014.zip',
            'http://images.cocodataset.org/zips/test2014.zip',
            'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',  # noqa
            'http://images.cocodataset.org/annotations/image_info_test2014.zip'  # noqa
        ],

        refcoco=[
            # images
            'http://images.cocodataset.org/zips/train2014.zip',
            # refcoco annotations
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip',
            # refcoco+ annotations
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip',
            # refcocog annotations
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip'
        ])
    url = data2url.get(args.dataset_name, None)

    if url is None:
        raise ValueError(f'Invalid dataset name {args.dataset_name}')

    download(
        url,
        dir=path,
        unzip=args.unzip,
        delete=args.delete,
        threads=args.threads)
