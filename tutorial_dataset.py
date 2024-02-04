import json
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)    #读取
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
class MIMICDataset(Dataset):
    def __init__(self):
        self.image_size = 512
        self.cond_type = "edge"                                
        self.cond_dir = '/hhd/1/dkm/KAD/CheXpert-v1.0-small/train_sketchs'
        self.image_dir = '/hhd/1/dkm/KAD/CheXpert-v1.0-small/train'
        if self.image_dir == '/hhd/1/dkm/KAD/CheXpert-v1.0-small/train':
            self.split = 'train'
        elif self.image_dir == '/hhd/1/dkm/KAD/CheXpert-v1.0-small/valid':
            self.split = 'valid'
         
        # self.split = 'valid2'
        data = pd.read_csv(self.split+'.csv', encoding='gbk')
        self.image_paths = ["/hhd/1/dkm/KAD/"+item for item in data['Path'].values.tolist()]
        labels = data[['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
                      'Pleural Effusion','Pleural Other','No Finding','Pneumothorax','Pneumonia','Support Devices']]
        labels2 = ["atelectasis", "cardiomegaly","consolidation","edema","enlargedcardiomediastinum","fracture","lunglesion",
        "lungopacity", "pleuraleffusion","pleuralother","nofinding","pneumothorax","pneumonia","supportdevices"]
        arr = labels.values.astype(float)
        
        self.label_list = []
        for j,item in enumerate(arr):
            line = []
            if item.sum() == 0:
                item[10] = 1
            index = np.where(item==1.0)
            for id in  index[0]:
                line.append(labels2[id])
            line_str = "_".join(line)
            self.label_list.append(line_str)
        
        # for i,item in enumerate(self.image_paths):
        #     path = "/hhd/1/dkm/lung-segmentation/real_valid3/"+item.split('valid/')[1].replace('/','_')[:-4]+'_'+label_list[i]+'.jpg'
        #     img = cv2.imread(item)
        #     img = cv2.resize(img,(512,512))
        #     cv2.imwrite(path,img)
            
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        
        target = cv2.imread(self.image_paths[idx])
        label = self.label_list[idx]
        source = cv2.imread(self.image_paths[idx].replace(self.split,self.split+'_sketchs'))
        prompt = "a xray with {}".format(label)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.resize(source, (self.image_size, self.image_size))
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        
        
        target = cv2.resize(target, (self.image_size, self.image_size))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = (target.astype(np.float32) / 127.5) - 1.0



        # Normalize target images to [-1, 1].

        return dict(jpg=target, txt=prompt, hint=source)

