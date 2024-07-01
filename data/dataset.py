from torch.utils import data
from torchvision.transforms import transforms as T
from PIL import Image
import csv

class CNNDataset(data.Dataset):
    def __init__(self, path_and_label_csv, train=True):
        super(CNNDataset, self).__init__()
        self.train = train
        csv_reader = csv.reader(open(path_and_label_csv))
        next(csv_reader)
        self.csv_rows = [row for row in csv_reader]
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])        

        if self.train:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path = self.csv_rows[index][0]
        label = int(self.csv_rows[index][1])
        
        data = Image.open(img_path)
        data = self.transforms(data)
        if self.train:
            return data, label
        else:
            return data, img_path, label

    def __len__(self):
        return len(self.csv_rows)
    
    
