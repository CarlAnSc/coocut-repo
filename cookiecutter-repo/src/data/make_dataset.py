## -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import TensorDataset
from torchvision import transforms


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    trains = [
        np.load(f"{input_filepath}/train_0.npz", mmap_mode="r"),
        np.load(f"{input_filepath}/train_1.npz", mmap_mode="r"),
        np.load(f"{input_filepath}/train_2.npz", mmap_mode="r"),
        np.load(f"{input_filepath}/train_3.npz", mmap_mode="r"),
        np.load(f"{input_filepath}/train_4.npz", mmap_mode="r")
    ]

    images, labels = [], []
    for x in trains:
        images.append(x["images"])
        labels.append(x["labels"])

    train_images, train_labels = np.concatenate(images), np.concatenate(labels)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0,), (1,)),
                                    transforms.Lambda(lambda x: torch.flatten(torch.swapdims(x, 0, 1), start_dim=1)),
                                    transforms.Lambda(lambda x: x.to(torch.float32))])

    train_set = {"images": transform(train_images), "labels": torch.LongTensor(train_labels)}

    test = np.load(f"{input_filepath}/test.npz", mmap_mode="r")

    test_set = {"images": transform(test['images']), "labels": torch.LongTensor(test['labels'])}

    torch.save(train_set, f=f"{output_filepath}/train.data")
    torch.save(test_set, f=f"{output_filepath}/test.data")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
