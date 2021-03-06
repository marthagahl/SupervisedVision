import os
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import argparse
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import glob
import json
import torchvision.transforms.functional as TF
from skimage.transform import rotate
import tarfile
import sys

import pathlib
from PIL import Image

#from retina_transform import foveat_img

from dataset import TransformedData


class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.resnet_model = torchvision.models.resnet50(pretrained = False)
        self.model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        self.fc1 = nn.Linear(2048,1000)
        self.fc2 = nn.Linear(1000,num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.type(torch.cuda.FloatTensor)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        cer = nn.CrossEntropyLoss()
        loss = cer(output, target)
        temp_loss = loss.detach().cpu().numpy()
        losses.append(temp_loss)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))

    losses = np.array(losses)
    average_loss = np.mean(losses)
    return average_loss, output


def test(args, model, device, test_loader, num_classes):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.type(torch.cuda.FloatTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            cer = nn.CrossEntropyLoss()
            test_loss += cer(output, target).detach().cpu().numpy()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
           
    test_loss /= len(test_loader)
    print ('correct:', correct)
    
    return test_loss, correct * 1.0 / (len(test_loader.dataset)) 


def main():
    parser = argparse.ArgumentParser(description = 'PyTorch Example')
    parser.add_argument('--batch_size', type = int, default = 48, metavar = 'N',
            help = 'input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type = int, default = 48, metavar = 'N',
            help = 'input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type = int, default = 40, metavar = 'N',
            help = 'number of epochs to train (default: 10)')
    parser.add_argument('--initial_lr', type = float, default = 0.0001, metavar = 'LR',
            help = 'learning rate (default: 0.01)')
    parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'M',
            help = 'SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action = 'store_true', default = False,
            help = 'disables CUDA training')
    parser.add_argument('--seed', type = int, default = 1, metavar = 'S',
            help = 'random seed (default: 1)')
    parser.add_argument('--log-interval', type = int, default = 5, metavar = 'N',
            help = 'how many batches to wait before logging training status')
    parser.add_argument('--classes', type = int, default = 16, metavar = 'N',
            help = 'number of classes')
    parser.add_argument('--workers', type = int, default = 4, metavar = 'N',
            help = 'number of workers')

    #Transformations
    parser.add_argument('--log_polar', default=False, action='store_true')
    parser.add_argument('--no-log_polar', dest='log_polar', action='store_false')
    parser.add_argument('--lp_out_shape', type = int, default = None, nargs = '*', metavar = 'N',
            help = 'output shape of log polar function')
    parser.add_argument('--salience_points', type = int, default = 1, metavar = 'N',
            help = 'number of points to sample')
    parser.add_argument('--training_aug', type = str, default = None,
            help = 'type of training augmentation')

    #required
    parser.add_argument('--dataset_path', type = str, required = True,
            help = 'path to dataset')
    parser.add_argument('--out_path', type = str, required = True,
            help = 'path to store experiments')
    parser.add_argument('--experiment_name', type = str, required = True,
            help = 'name for experiment')



    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

#    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

    id_nums = [4, 8, 16, 32, 64]
#    id_nums = [32]

    outpath = '{}/{}'.format(args.out_path, args.experiment_name)
    os.makedirs(outpath, exist_ok = True)


    num_classes = args.classes
    if args.lp_out_shape is not None:
        out_shape = tuple(args.lp_out_shape)
    else:
        out_shape = None

    model = Model(num_classes)
    model = nn.DataParallel(model).cuda()

    low_val_upright_loss = np.inf

    optimizer =  optim.Adam(list(model.parameters()), lr = args.initial_lr, weight_decay = 1e-3)

    loss_file = open(f'{outpath}/loss.txt', 'w')
    
    for val in id_nums:

        print ("\n Training with {} identities \n".format(val))
        ### Data loaders
        train_dataset = TransformedData("{}/{}_identities/train/".format(args.dataset_path, val), "",  180, 15, args.log_polar, out_shape, augmentation = args.training_aug, points = args.salience_points, inversion = False)
        train_dataset_acc = TransformedData("{}/{}_identities/train/".format(args.dataset_path, val), "",  180, 0, args.log_polar, out_shape, augmentation = None, points = args.salience_points, inversion = False)
        val_dataset_1 = TransformedData("{}/{}_identities/valid/".format(args.dataset_path, val), "",  180, 0, args.log_polar, out_shape, augmentation = None, points = args.salience_points, inversion = False)
        val_dataset_2 = TransformedData("{}/{}_identities/valid/".format(args.dataset_path, val), "",  180, 0, args.log_polar, out_shape, augmentation = None, points = args.salience_points, inversion = True)
        test_dataset_1 = TransformedData("{}/{}_identities/test/".format(args.dataset_path, val), "",  180, 0, args.log_polar, out_shape, augmentation = None, points = args.salience_points, inversion = False)
        test_dataset_2 = TransformedData("{}/{}_identities/test/".format(args.dataset_path, val), "",  180, 0, args.log_polar, out_shape, augmentation = None, points = args.salience_points, inversion = True)


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle = True, 
            )
        train_loader_acc = torch.utils.data.DataLoader(
            train_dataset_acc,
            batch_size=args.batch_size,
            shuffle = True, 
            )
        val_loader_1 = torch.utils.data.DataLoader(
            val_dataset_1,
            batch_size=args.batch_size,
            shuffle = True, 
            )
        val_loader_2 = torch.utils.data.DataLoader(
            val_dataset_2,
            batch_size=args.batch_size,
            shuffle = True, 
            )
        test_loader_1 = torch.utils.data.DataLoader(
            test_dataset_1,
            batch_size=args.batch_size,
            shuffle = True, 
            )
        test_loader_2 = torch.utils.data.DataLoader(
            test_dataset_2,
            batch_size=args.batch_size,
            shuffle = True, 
            )


        for epoch in range(0, args.epochs):
            train(args, model, device, train_loader, optimizer, epoch)
            
            print("\nEvaluating on training set...")
            train_loss, train_acc = test(args, model, device, train_loader_acc, num_classes)
            print("\nEvaluating on upright validation set...")
            val_upright_loss, val_upright_acc = test(args, model, device, val_loader_1, num_classes)
            print("\nEvaluating on inverted validation set...")
            val_inverted_loss, val_inverted_acc = test(args, model, device, val_loader_2, num_classes)


            print(f'\nEpoch {epoch} Training Set Loss: {train_loss}')
            print(f'Epoch {epoch} Training Set Accuracy: {train_acc}')
            print(f'Epoch {epoch} Upright Validation Loss: {val_upright_loss}')
            print(f'Epoch {epoch} Upright Validation Accuracy: {val_upright_acc}')
            print(f'Epoch {epoch} Inverted Validation Loss: {val_inverted_loss}')
            print(f'Epoch {epoch} Inverted Validation Accuracy: {val_inverted_acc}')

            # save best model
            if val_upright_loss < low_val_upright_loss:
                print("Saving new best model...")
                low_val_upright_loss = val_upright_loss
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    },
                    f'{outpath}/best_model.pth'
                )


            json_data = {'train loss': str(train_loss), 
                         'train accuracy': str(train_acc), 
                         'val upright loss': str(val_upright_loss),
                         'val upright accuracy': str(val_upright_acc),
                         'val inverted loss': str(val_inverted_loss),
                         'val inverted accuracy': str(val_inverted_acc)
            }   
           
            with open(f'{outpath}/metrics_{epoch}.json','w') as f:
                json.dump(json_data, f)


            loss_data = "epoch{}:{},{},{},{},{},{}\n".format(epoch,train_loss,train_acc,val_upright_loss,val_upright_acc,val_inverted_loss,val_inverted_acc)
            loss_file.write(loss_data)


    loss_file.close()


    print()
    print("Loading best model...")
    ### Load previously saved weights
    checkpoint = torch.load(f'{outpath}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}...")


    print("\nEvaluating on upright test set...")
    test_upright_loss, test_upright_acc = test(args, model, device, test_loader_1, num_classes)
    print("\nEvaluating on inverted test set...")
    test_inverted_loss, test_inverted_acc = test(args, model, device, test_loader_2, num_classes)

    print(f'\nUpright Test Set Loss: {test_upright_loss}')
    print(f'Upright Test Set Accuracy: {test_upright_acc}')
    print(f'\nInverted Test Set Loss: {test_inverted_loss}')
    print(f'Inverted Test Set Accuracy: {test_inverted_acc}')


if __name__ == '__main__':
    main()

