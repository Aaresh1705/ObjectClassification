import os
import glob
import PIL.Image as Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class loadDataset(Dataset):
    def __init__(self, transform, data_path='/'):
        self.transform = transform
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path + '/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.png')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert('RGB') # Use .convert('RGB') for consistent loading
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


def dataset(batch_size=64, transform=None):
    if transform is None:
        transform = []

    size = 128
    train_transform = transforms.Compose([transforms.Resize((size, size)),
                                          *transform,
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((size, size)),
                                         transforms.ToTensor()])

    full_dataset = loadDataset(transform=train_transform)

    trainset = loadDataset(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    testset = loadDataset(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

    return (train_loader, test_loader), (trainset, testset)

def datasetPotholes(batch_size=8, transform=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    dataset = loadDataset(transform=train_transform)

    total_len = len(dataset)
    train_len = max(1, int(total_len * train_ratio))
    val_len = max(0, int(total_len * val_ratio))
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader), (train_set, val_set, test_set)