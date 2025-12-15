import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from lama_cleaner.model_manager import ModelManager
from PIL import Image
import cv2
import numpy as np

class WatermarkDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*/*.jpg')))
        self.annotation_paths = sorted(glob.glob(os.path.join(root_dir, '*/*.txt')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        annot_path = self.annotation_paths[index]
        
        # Читаем изображение и аннотацию
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotation = np.loadtxt(annot_path)
        
        if self.transforms:
            transformed = self.transforms(image=img, bboxes=annotation)
            img = transformed["image"]
            annotation = transformed["bboxes"]
        
        return img, annotation

class WatermarkDetection(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.net = ModelManager(name="lama", device=get_device())  # LAMA cleaner
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        imgs, annotations = batch
        preds = self.forward(imgs)
        loss = self.criterion(preds, annotations)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, annotations = batch
        preds = self.forward(imgs)
        val_loss = self.criterion(preds, annotations)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Watermark Detection")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--learning_rate", "--lr", dest="lr", type=float, default=1e-3)
        return parser
