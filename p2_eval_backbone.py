import torch
from byol_pytorch import BYOL
from torchvision import models
from p2_datasets import *
from torch.utils.data import DataLoader
import json
from torchvision.transforms import transforms
from tqdm import tqdm

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

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

train_tfm = transforms.Compose([
    ## TO DO ##
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])


if __name__ == '__main__':
    fixed_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_data_path = './hw4_data/mini/train/'
    training_csv_path = './hw4_data/mini/train.csv'
    batch_size = 4
    num_epoch = 500
    # ckpt_save_path = './byol_ckpt_new'
    # os.makedirs(ckpt_save_path, exist_ok=True)
    with open('./mini_translate.json', 'r') as f:
        mini_translate_dict = json.load(f)
    train_set = CustomImageDataset(data_folder_path=training_data_path, csv_file_path=training_csv_path, translate_dict=mini_translate_dict, have_label=True, transform=train_tfm)
    # val_set = CustomImageDataset(data_folder_path=val_data_path, have_label=True, transform=val_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    resnet = models.resnet50(weights=None)
    load_parameters(resnet, '/home/hsien/dlcv/hw4-ehsienmu/byol_ckpt/byol_epoch_44.pt')
    resnet = resnet.to(device)
    # learner = BYOL(
    #     resnet,
    #     image_size=128,
    #     hidden_layer='avgpool'
    # )

    # opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


    # def sample_unlabelled_images():
    #     return torch.randn(20, 3, 256, 256)
    corr_num = 0
    item_cnt = 0
    # for ep in range(num_epoch):
    for batch_idx, ( images, label,) in enumerate(tqdm(train_loader)):
    # for _ in range(100):
        # images = sample_unlabelled_images()
        
        images = images.to(device)
        label = label.to(device)
        output = resnet(images) 
        
        pred = output.argmax(dim=1)
        
        # correct if label == predict_label
        corr_num += (pred.eq(label.view_as(pred)).sum().item())
        item_cnt += len(label)
        print(corr_num, item_cnt)
    train_acc = corr_num / len(train_loader.dataset)
    print('train_accuracy:', train_acc)

        # loss = learner(images)
        # opt.zero_grad()
        # loss.backward()
        # opt.step()
        # learner.update_moving_average()  # update moving average of target encoder

    # save your improved network
    # torch.save(resnet.state_dict(), './byol.pt')
    # torch.save(resnet.state_dict(), os.path.join(ckpt_save_path, f'byol_epoch_{ep}.pt'))
