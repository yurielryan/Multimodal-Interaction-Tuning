import random
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torch import nn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # related with operation speed
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.set_default_dtype(torch.float64)


def save_checkpoint(save_file_path, epoch, video_model, optimizer, scheduler):
    
    if hasattr(video_model, 'module'):
        video_model_state_dict = video_model.module.state_dict()
    else:
        video_model_state_dict = video_model.state_dict()
    
    save_states = {
        'epoch': epoch,
        'state_dict': video_model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(save_states, save_file_path)

def resume_model(resume_path,  model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    return model

class cls_network(nn.Module):
    def __init__(self, input_dim, hidden_dim = 32, output_dim = 2):
        super(cls_network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
       )
    def forward(self, x):
        x = self.fc(x)
        return x


class feature_dataset(Dataset):
    
    def __init__(self, modality1, modality2, target):
        self.modality1 = modality1
        self.modality2 = modality2
        self.target = target
        
    def __len__(self):
        "Returns the total number of samples."
        return self.target.size(0)
    
    def __getitem__(self, index): 
       return self.modality1[index], self.modality2[index], self.target[index] 
   
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def get_loader(opt, data_dir, shuffle=True):
    data = torch.load(data_dir)
    train_data = feature_dataset(data['train_modal_1_features'], data['train_modal_2_features'], data['train_targets'])  
    val_data = feature_dataset(data['val_modal_1_features'], data['val_modal_2_features'], data['val_targets'])

    g = torch.Generator()
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=getattr(opt, 'num_workers', 4),
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=getattr(opt, 'num_workers', 4),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # Optional test loader if test split present
    test_loader = None
    if all(k in data for k in (
        'test_modal_1_features', 'test_modal_2_features', 'test_targets'
    )):
        test_data = feature_dataset(
            data['test_modal_1_features'], data['test_modal_2_features'], data['test_targets']
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=getattr(opt, 'num_workers', 4),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g,
        )

    return (train_loader, val_loader, test_loader) if test_loader is not None else (train_loader, val_loader)


# infer input dims
def infer_input_dims_from_pt(pt_path, cfg):
    data = torch.load(pt_path, map_location='cpu')
    if 'train_modal_1_features' in data and 'train_modal_2_features' in data:
        cfg.input_size_1 = int(data['train_modal_1_features'].shape[1])
        cfg.input_size_2 = int(data['train_modal_2_features'].shape[1])
        
def obtain_feature_input(batch, device):
    modal_1 = batch[0].to(device)
    modal_2 = batch[1].to(device)
    labels = batch[2].to(device)
    return modal_1, modal_2, labels
