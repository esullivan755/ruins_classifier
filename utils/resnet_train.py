# Utils for training ResNet Encoder from tensor batches

# Imports

#Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import resnet50
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Plotting/Utils
from tqdm import tqdm
import matplotlib.pyplot as plt


# Dataset Class
class Tensor_Dataset(Dataset):
    def __init__(self,data,labels=None,coord=None,transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        dset = self.data[idx]
        if self.labels is not None:
            lab = self.labels[idx]
            return dset, lab
        if self.transform:
            dset = self.transform(dset)
        return dset


# Plotting
def display_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Displays model performance plots

    train_acc, valid_acc, train_loss, valid_loss: lists of epoch, acc/loss
    """

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='tab:blue', linestyle='-', label='train loss')
    plt.plot(valid_loss, color='tab:red', linestyle='-', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='tab:blue', linestyle='-', label='train accuracy')
    plt.plot(valid_acc, color='tab:red', linestyle='-', label='validataion accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Training
def build_resnet(input_channels=6,device='cuda',optimizer='SGD',lr=1e-2):
    """
    Builds a resnet feature extractor based on the number of input channels

    input_channels: int
    """
    assert optimizer in ['SGD','Adam'], f'Optimizer type {optimzer} not supported.'
    device = torch.device(device)
    model = models.resnet50(pretrained = True)
    if input_channels != 3:

        #Pretrained resnet weights from torchvision
        pretrained_weights = model.conv1.weight.clone()

        #Replacing initial layer with new conv1, modifying input channels for this dataset
        model.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        #Initialize the new conv1 weights using pretrained weights
        with torch.no_grad():
            model.conv1.weight[:, :3] = pretrained_weights  #RGB
            model.conv1.weight[:, 3:] = pretrained_weights[:, :3] / 3.0  #NDVI, NDBI, Elevation

    num_classes = 2 #yes/no image is a ruin
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features,num_classes)
    model = model.to(device)

    #Adam or SGD+Momentum
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9)
        scheduler = False

    else:

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)


    criterion = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion, device



def train_resnet(model,optimizer,criterion,device,dataloader):
    """
    resnet training loop, forward and backward pass with weights update

    model: torch model
    optimizer: torch optimizer
    criterion: torch loss function, progress scoring method
    """
    model.train()
    total_running_loss = 0
    train_running_correct = 0
    print('Training')

    for x,y in tqdm(dataloader):

        x,y = x.to(device),y.long().to(device)
        #reset gradients
        optimizer.zero_grad()

        #forward pass
        output = model(x)

        #calculating loss
        loss = criterion(output,y)

        total_running_loss += loss.item()

        _,preds = torch.max(output.data,1)

        #correct class predictions
        train_running_correct += (preds==y).sum().item()

        # Calculate new weights
        loss.backward()

        # Apply new weights
        optimizer.step()



    train_loss = total_running_loss/len(dataloader)
    train_acc =  100.*train_running_correct/len(dataloader.dataset)

    return train_loss, train_acc


def val_resnet(model,optimizer,criterion,device,dataloader,cm=False):
    """
    Validating loop, scoring & performance

    model: torch model
    optimizer: torch optimizer
    criterion: torch loss function, progress scoring method
    """

    model.eval()
    print('Validating')
    total_running_loss = 0
    val_running_correct = 0
    cm_labels = []
    cm_preds = []

    with torch.no_grad():

        for x,y in tqdm(dataloader):

            x,y = x.to(device),y.long().to(device)

            #forward pass
            output = model(x).to(device)

            loss = criterion(output,y)

            total_running_loss += loss.item()

            _,preds = torch.max(output.data,1)

            val_running_correct += (preds==y).sum().item()

            #for final test set:
            if cm:
                cm_preds.extend(preds.cpu().numpy())
                cm_labels.extend(y.cpu().numpy())


    val_loss = total_running_loss/len(dataloader)
    val_acc =  100.*val_running_correct/len(dataloader.dataset)

    #for final test set:
    if cm:
        cm = confusion_matrix(cm_labels,cm_preds)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        return val_loss, val_acc, display
    else:
        return val_loss, val_acc


def prepare_feature_extractor_from_state_dict(state_dict, device='cuda'):
    """
    Creating the feature extractor based on the previously trained weights

    state_dict: Previsouly trained weights, torch tensor.
    device: device the environment is running on, str.
    """
    model = models.resnet50(pretrained=False)


    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)


    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    # Loading saved weights
    model.load_state_dict(state_dict)

    # Removing final classification layer to make it a feature extractor
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor



def extract_features(feature_extractor, dataloader, device='cuda'):
    """
    Function to extraxt (N,2048) features from (Num_Batches, Batch_Size, 6, 224, 224) dataset

    feature_extractor: a torch model object

    dataloader: the dataset, torch dataloader object
    """
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:

            #For unlabeled data (potential ruins)
            if isinstance(batch, (tuple, list)) and len(batch) == 1:
                x = batch[0]
                y = None
            #For labeled datasets x & y with or without the coords
            elif isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]

            else:
                x = batch
                y = None

            if x.dim() == 3:
                x = x.unsqueeze(0)

            x = x.to(device)
            feats = feature_extractor(x)          # [B, 2048, 1, 1]
            feats = feats.view(feats.size(0), -1) #Flattened to [B, 2048]
            features_list.append(feats.cpu())

            if y is not None:
                y = y.cpu()
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                labels_list.append(y)


    if features_list and labels_list:
        return torch.cat(features_list), torch.cat(labels_list)
    elif features_list:
        return torch.cat(features_list), None
    else:
        #Return None if blank features get passed through
        return None, None
