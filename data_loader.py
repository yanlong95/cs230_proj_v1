import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image



"""
The file help to read the image data
args: path
"""

# train transformer
train_transformer = transforms.Compose([
    transforms.Resize(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

# evl and test transformer
eval_transformer = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor()])


# Dataloader
class concreteDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames]
        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]


# load a train, val, text in mini-batch size
def fetch_dataloader(types, data_dir, params, device='cpu'):
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_mix".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(concreteDataset(path, train_transformer), **params)
            else:
                dl = DataLoader(concreteDataset(path, eval_transformer), **params)

            dataloaders[split] = dl

    return dataloaders

# # test
# def load_data(path):
#     return concreteDataset(path, train_transformer)
