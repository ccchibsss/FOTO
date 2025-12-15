import os
import random
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare_dataset(data_root: str, split_ratio: float = 0.8):
    """Разделение данных на тренировочные и проверочные наборы."""
    images_folder = os.path.join(data_root, 'images')
    annotations_folder = os.path.join(data_root, 'annotations')
    
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    annotation_files = [os.path.splitext(f)[0]+'.txt' for f in image_files]
    
    train_images, val_images = train_test_split(image_files, test_size=split_ratio)
    train_annotations = [os.path.splitext(f)[0]+'.txt' for f in train_images]
    val_annotations = [os.path.splitext(f)[0]+'.txt' for f in val_images]
    
    # Копируем файлы в нужные директории
    create_directory_structure(train_images, train_annotations, 'train')
    create_directory_structure(val_images, val_annotations, 'validation')

def create_directory_structure(images, annotations, subset_name):
    base_dir = os.path.join('datasets', subset_name)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'annotations'), exist_ok=True)
    
    for img, ann in zip(tqdm(images), annotations):
        shutil.copyfile(os.path.join('data/images', img), os.path.join(base_dir, 'images', img))
        shutil.copyfile(os.path.join('data/annotations', ann), os.path.join(base_dir, 'annotations', ann))

prepare_dataset('data')
