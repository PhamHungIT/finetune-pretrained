import argparse
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import utils
from data.preprocess import clean_text

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/data_demo', 
                    help="Directory containing the dataset")
parser.add_argument('--data_name', help='Name file containing raw data')
parser.add_argument('--col_text', help='Column containing text')
parser.add_argument('--col_category', help='Column containing categories respectively')
parser.add_argument('--split_type', default='train')
parser.add_argument('--split_file', action='store_true', help='Optional split file')


def save_dataset(data: dict, save_dir):
    """Writes text.txt and category.txt files in save_dir from dataset

    Args:
        dataset: [("iphnone 14 promax", "Điện tử - Điện máy"), ...]
        save_dir: (string)
    """

    # Create directory if it doesn't exist
    print("\nSaving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'text.txt'), 'w') as file_text:
        file_text.write("\n".join(data['text']))
    
    with open(os.path.join(save_dir, 'category.txt'), 'w') as file_category:
        file_category.write("\n".join(data['category']))
    print("- Done!")


if __name__ == "__main__":
    args = parser.parse_args()

    path_dataset = os.path.join(args.data_dir, args.data_name)
    msg = "{} file not found. Make sure you have downloaded \
        the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("\n\nLoading csv dataset into memory...")
    raw_data = utils.load_csv(
        csv_file=path_dataset,
        col_text=args.col_text,
        col_category=args.col_category,
        sep=','
    )
    print("- Done!")

    print('\nCleaning data...')
    raw_data['text'] = list(map(
        clean_text,
        tqdm(raw_data['text']))
    )
    print("- Done!")
    
    """Split the dataset into train, val and test
    (dummy split with no shuffle)"""
    if args.split_file:
        train_text, val_text, train_category, val_category = train_test_split(
            raw_data['text'],
            raw_data['category'],
            test_size=0.2,
            random_state=100
        )

        train_data = {
            'text': train_text,
            'category': train_category
        }

        val_data = {
            'text': val_text,
            'category': val_category
        }

    # Save the datasets to files
        save_dataset(train_data, os.path.join(args.data_dir, 'train'))
        save_dataset(val_data, os.path.join(args.data_dir, 'val'))

    else:

        data = {
            'text': raw_data['text'],
            'category': raw_data['category']
        }
        save_dataset(data, os.path.join(args.data_dir, args.split_type))