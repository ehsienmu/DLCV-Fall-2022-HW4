import torch
from byol_pytorch import BYOL
from torchvision import models
from p2_datasets import *
from torch.utils.data import DataLoader
import json
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn as nn
import time
import torch.optim as optim 
import os
import argparse

def fixed_seed(myseed):
    # np.random.seed(myseed)
    # random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
        
def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)#, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")

def train(model, train_loader, val_loader, num_epoch, early_stop, log_path, save_path, device, criterion, scheduler, optimizer):
    start_train = time.time()
    # overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    # overall_acc = np.zeros(num_epoch ,dtype = np.float32)
    # overall_val_loss = np.zeros(num_epoch ,dtype=np.float32)
    # overall_val_acc = np.zeros(num_epoch ,dtype = np.float32)

    overall_loss = [] #np.zeros(num_epoch ,dtype=np.float32)
    overall_acc = [] #np.zeros(num_epoch ,dtype = np.float32)
    overall_val_loss = [] #np.zeros(num_epoch ,dtype=np.float32)
    overall_val_acc = [] #np.zeros(num_epoch ,dtype = np.float32)

    best_acc = 0.0
    last_val_acc = 0.0
    last_last_val_acc = 0.0
    early_stop_cnt = 0
    for i in range(num_epoch):
        print(f'epoch = {i}')
        # epcoch setting
        start_time = time.time()
        train_loss = 0.0 
        corr_num = 0
        train_acc = 0.0


        # training part
        # start training
        model.train()
        print(len(train_loader))
        for batch_idx, ( _, data, label,) in enumerate(tqdm(train_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            label = label.to(device)

            # pass forward function define in the model and get output 
            output = model(data) 

            # calculate the loss between output and ground truth
            loss = criterion(output, label)
            
            # discard the gradient left from former iteration 
            optimizer.zero_grad()

            # calcualte the gradient from the loss function 
            loss.backward()
            
            # if the gradient is too large, we dont adopt it
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # Update the parameters according to the gradient we calculated
            optimizer.step()

            train_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            
            # correct if label == predict_label
            corr_num += (pred.eq(label.view_as(pred)).sum().item())

        # scheduler += 1 for adjusting learning rate later
        # scheduler.step()
        
        # averaging training_loss and calculate accuracy
        train_loss = train_loss / len(train_loader.dataset) 
        train_acc = corr_num / len(train_loader.dataset)
                
        # record the training loss/acc
        # overall_loss[i], overall_acc[i] = train_loss, train_acc
        overall_loss.append(train_loss)
        overall_acc.append(train_acc)

        #############
        ## TO DO ##
        # validation part 
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            corr_num = 0
            val_acc = 0.0
            
            ## TO DO ## 
            # Finish forward part in validation. You can refer to the training part 
            # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 

            for batch_idx, (_, data, label,) in enumerate(tqdm(val_loader)):
                # put the data and label on the device
                # note size of data (B,C,H,W) --> B is the batch size
                data = data.to(device)
                label = label.to(device)

                # pass forward function define in the model and get output 
                output = model(data) 

                # calculate the loss between output and ground truth
                loss = criterion(output, label)
                
                # # discard the gradient left from former iteration 
                # optimizer.zero_grad()

                # # calcualte the gradient from the loss function 
                # loss.backward()
                
                # # if the gradient is too large, we dont adopt it
                # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
                
                # # Update the parameters according to the gradient we calculated
                # optimizer.step()

                val_loss += loss.item()

                # predict the label from the last layers' output. Choose index with the biggest probability 
                pred = output.argmax(dim=1)
                
                # correct if label == predict_label
                corr_num += (pred.eq(label.view_as(pred)).sum().item())

            # scheduler += 1 for adjusting learning rate later
            
            # averaging training_loss and calculate accuracy
            val_loss = val_loss / len(val_loader.dataset) 
            val_acc = corr_num / len(val_loader.dataset)
            last_val_acc = val_acc        
            # record the training loss/acc
            # overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc
            overall_val_loss.append(val_loss)
            overall_val_acc.append(val_acc)
            scheduler.step(val_loss)
            # scheduler.step()
        #####################
        
        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        #print(f'epoch = {i}')
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        print('========================\n')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {i}\n', )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        # save model for every epoch 
        torch.save(model.state_dict(), os.path.join(save_path, f'resnet_epoch_{i}.pt'))
        early_stop_cnt += 1
        # save the best model if it gain performance on validation set
        if  val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_resnet_model.pt'))
            early_stop_cnt = 0
        if early_stop_cnt > early_stop:
            print('early stop!')
            break
        

    

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

tfm = transforms.Compose([
    ## TO DO ##
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--freeze', default=False, type=bool)
    parser.add_argument('--backbone_filename', default='', type=str)
    parser.add_argument('--save_dir_name', default='', type=str)
    # parser.add_argument('--input_csv_file', default='', type=str)
    # parser.add_argument('--output_file', default='', type=str)
    # parser.add_argument('--model_file', default='', type=str)

    args = parser.parse_args()
    fixed_seed(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_data_path = './hw4_data/office/train/'
    training_csv_path = './hw4_data/office/train.csv'
    val_data_path = './hw4_data/office/val/'
    val_csv_path = './hw4_data/office/val.csv'
    backbone_filename = args.backbone_filename # '/home/hsien/dlcv/hw4-ehsienmu/byol_ckpt/byol_epoch_62.pt'
    with open('./office_translate.json', 'r') as f:
        office_translate_dict = json.load(f)
    
    batch_size = 4
    num_epoch = 500
    early_stop = 30

    model_name = args.save_dir_name # 'ssl_resnet'
    ckpt_save_path = './' + model_name + '_ckpt'
    os.makedirs('./acc_log', exist_ok=True)
    os.makedirs(ckpt_save_path, exist_ok=True)
    log_path = os.path.join('./acc_log', 'acc_' + model_name + '_.log')

    train_set = CustomImageDataset(data_folder_path=training_data_path, csv_file_path=training_csv_path, translate_dict=office_translate_dict, have_label=True, transform=tfm)
    val_set = CustomImageDataset(data_folder_path=val_data_path, csv_file_path=val_csv_path, translate_dict=office_translate_dict, have_label=True, transform=tfm)
    # val_set = CustomImageDataset(data_folder_path=val_data_path, have_label=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    resnet = models.resnet50(weights=None)
    load_parameters(resnet, backbone_filename)
    resnet.fc = nn.Sequential(nn.Linear(2048, 65))

    resnet = resnet.to(device)

    # learner = BYOL(
    #     resnet,
    #     image_size=128,
    #     hidden_layer='avgpool'
    # )

    optimizer = torch.optim.Adam(resnet.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, cooldown=3)

    criterion = nn.CrossEntropyLoss()
   
    train(model=resnet, train_loader=train_loader, val_loader=val_loader, 
        num_epoch=num_epoch, early_stop=early_stop, log_path=log_path, save_path=ckpt_save_path,
        device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    # def sample_unlabelled_images():
    #     return torch.randn(20, 3, 256, 256)

    # for ep in range(num_epoch):
    #     print(f'epoch = {ep}')
    #     train_loss = 0.0 
    #     corr_num = 0
    #     train_acc = 0.0
    #     resnet.train()
    #     for batch_idx, ( images, label,) in enumerate(tqdm(train_loader)):
    #     # for _ in range(100):
    #         # images = sample_unlabelled_images()
            
    #         images = images.to(device)
    #         # label = label.to(device)
    #         output = resnet(images) 
    #         loss = criterion(output, label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # learner.update_moving_average()  # update moving average of target encoder
    #         train_loss += loss.item()
    #         pred = output.argmax(dim=1)
    #         corr_num += (pred.eq(label.view_as(pred)).sum().item())
        
    #     train_loss = train_loss / len(train_loader.dataset) 
    #     train_acc = corr_num / len(train_loader.dataset)
        
    #     with torch.no_grad():
    #         resnet.eval()
    #         val_loss = 0.0
    #         corr_num = 0
    #         val_acc = 0.0 
            
    #         for batch_idx, (data, label,) in enumerate(tqdm(val_loader)):

    #             data = data.to(device)
    #             label = label.to(device)

    #             # pass forward function define in the model and get output 
    #             output = resnet(data) 

    #             # calculate the loss between output and ground truth
    #             loss = criterion(output, label)
    #             val_loss += loss.item()

    #             # predict the label from the last layers' output. Choose index with the biggest probability 
    #             pred = output.argmax(dim=1)
                
    #             # correct if label == predict_label
    #             corr_num += (pred.eq(label.view_as(pred)).sum().item())
    #         val_loss = val_loss / len(val_loader.dataset) 
    #         val_acc = corr_num / len(val_loader.dataset)
    #         last_val_acc = val_acc        
    #         # record the training loss/acc
    #         # overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc
    #         overall_val_loss.append(val_loss)
    #         overall_val_acc.append(val_acc)
    #         scheduler.step(val_loss)
    #     # save your improved network
    #     # torch.save(resnet.state_dict(), './byol.pt')
    #     torch.save(resnet.state_dict(), os.path.join(ckpt_save_path, f'byol_epoch_{ep}.pt'))
