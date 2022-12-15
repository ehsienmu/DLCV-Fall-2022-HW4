import torch
import os
# import random

from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
from PIL import Image
from torchvision.datasets import DatasetFolder

from torch.utils.data.dataset import Dataset
from torchvision import models

# from p1_model_cfg import mycnn_cfg, pretrained_resnet50_cfg
import argparse

from torchvision.transforms import transforms
# from p1_datasets import *
import numpy as np
from tqdm import tqdm
# import argparse
# import glob
import json
import pandas as pd

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

tfm = transforms.Compose([
    ## TO DO ##
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])

def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)#, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")

def fixed_seed(myseed):
    np.random.seed(myseed)
    # random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
        

class CustomImageDataset(Dataset):
    def __init__(self, data_folder_path, csv_file_path, translate_dict, have_label, transform=None):
        data_df = pd.read_csv(csv_file_path)
        
        # path = r'./hw1_data/hw1_data/p1_data/train_50/*.png'
        # if (data_folder_path[-1] != '/'):
        #     data_folder_path += '/'
        # images_filename = glob.glob(data_folder_path+'*.png')

        # print(images_filename)
        if have_label:
            # labels = []
            # for full_path in images_filename:
            #     labels.append(int(full_path.split('/')[-1].split('_')[0]))
            # # print('labels[:5]:', labels[:5])
            labels = [translate_dict[x] for x in data_df['label']]
            self.labels = torch.tensor(labels)
        else:
            self.labels = None

        # It loads all the images' file name and correspoding labels here
        self.images = data_df['filename']# images_filename
        self.id_names = data_df['id']
        # The transform for the image
        self.transform = transform

        # prefix of the files' names
        self.prefix = data_folder_path

        # self.translate_label = translate_dict

        print('images from', self.prefix)
        # print('csv from', self.prefix)
        print(f'Number of images is {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform

        # You shall return image, label with type "long tensor" if it's training set
        # pass
        # full_path = os.path.join(self.prefix, self.images[idx])
        img = Image.open(os.path.join(self.prefix, self.images[idx])).convert("RGB")
        if self.transform is not None:
            transform_img = self.transform(img)

        if self.labels != None:
            #  print(type((transform_img, self.labels[idx])))
            return (self.id_names[idx], transform_img, self.labels[idx], self.images[idx])
        else:
            return (self.id_names[idx], transform_img, self.images[idx])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', default='', type=str)
    parser.add_argument('--input_csv_file', default='', type=str)
    parser.add_argument('--output_file', default='', type=str)
    parser.add_argument('--model_file', default='', type=str)
    
    args = parser.parse_args()

    
    with open('./office_translate.json', 'r') as f:
        office_translate_dict = json.load(f)

    # fixed random seed
    fixed_seed(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    """ training hyperparameter """

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Linear(2048, 65))
    load_parameters(model, args.model_file)
    # Put model's parameters on your device
    model = model.to(device)
    
    # print(model)
   
    # train_set = CustomImageDataset(data_folder_path=training_data_path, have_label=True, transform=train_tfm)
    # val_set = CustomImageDataset(data_folder_path=args.input_dir, have_label=True, transform=val_tfm)
    
    val_set = CustomImageDataset(data_folder_path=args.input_dir, csv_file_path=args.input_csv_file, translate_dict=office_translate_dict, have_label=True, transform=tfm)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # print(val_set)

    # sdfdf
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    # criterion = nn.CrossEntropyLoss()
    
    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    # count_parameters(model)
    results = []
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        corr_num = 0
        val_acc = 0.0
        
        ## TO DO ## 
        # Finish forward part in validation. You can refer to the training part 
        # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 

        for batch_idx, (_, data, label, fname,) in enumerate(tqdm(val_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            label = label.to(device)

            # pass forward function define in the model and get output 
            output = model(data) 

            # calculate the loss between output and ground truth
            # loss = criterion(output, label)
            
            # # discard the gradient left from former iteration 
            # optimizer.zero_grad()

            # # calcualte the gradient from the loss function 
            # loss.backward()
            
            # # if the gradient is too large, we dont adopt it
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # # Update the parameters according to the gradient we calculated
            # optimizer.step()

            # val_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            
            # results.append((fname[0].split('/')[-1], str(int(pred[0]))))
            
            # correct if label == predict_label
            corr_num += (pred.eq(label.view_as(pred)).sum().item())

        # scheduler += 1 for adjusting learning rate later
        
        # averaging training_loss and calculate accuracy
        # val_loss = val_loss / len(val_loader.dataset) 
        val_acc = corr_num / len(val_loader.dataset)
        print('val acc =', val_acc)
        # record the training loss/acc
        # overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc
    #     overall_val_loss.append(val_loss)
    #     overall_val_acc.append(val_acc)
    #     scheduler.step(val_loss)
    #     # scheduler.step()
    # #####################
        
        # Display the results
        
        # # print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        # print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        # print('========================\n')

    # with open(args.output_file, 'w') as f:
    #     f.write('filename,label\n')
    #     for fname, predl in results:
    #         f.write(fname)
    #         f.write(',')
    #         f.write(predl)
    #         f.write('\n')
        