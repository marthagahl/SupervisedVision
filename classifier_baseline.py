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

from dataset import TransformedData


class Flatten(nn.Module):
    def forward(self,input):
#        print (input.shape)
        return input.view(input.size(0), -1)


def smallArchitecture(num_classes, **kwargs):

    model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 7, stride = 2, padding = 3, bias = False ),
            nn.ReLU(),
#            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1),
            Flatten(),
            nn.Linear(11552, 40),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(40, num_classes),
            )

    return model


def largeArchitecture(num_classes, out_shape, **kwargs):

    if out_shape == (190, 165):
        in_size = 14720
    elif out_shape == (180, 180):
        in_size = 15488
    else:
        in_size = 21120
    model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 3, bias = False ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
#            nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = False),
            nn.ReLU(),
#            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1),
            Flatten(),
            nn.Linear(in_size, 40), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(40, num_classes),
            )

    return model


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
#        data = data.type(torch.cuda.FloatTensor)
        data = data.type(torch.FloatTensor)
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
    return average_loss


def test(args, model, device, test_loader, num_classes):
    model.eval()
    test_loss = 0
    correct = 0
    
    num_class = {}
    class_correct = {}
    top_17 = [338, 435, 573, 681, 769, 779, 785, 855, 906, 1052, 1088, 1122, 1605, 2120, 2173, 2198, 2265]

    with torch.no_grad():
        for data, target in test_loader:
#            data = data.type(torch.cuda.FloatTensor)
            data = data.type(torch.FloatTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            cer = nn.CrossEntropyLoss()
            test_loss += cer(output, target).detach().cpu().numpy()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


            for i in range(num_classes):
                if i in top_17:
                    class_indices = (target == i).nonzero()
                    num_class[i] = class_indices.shape[0]
                    class_correct[i] = (pred[class_indices] == i).long().sum().item()

    for i in range(num_classes):
        if i in top_17:
            try:
                print(f'Class {i} Accuracy: {class_correct[i] / num_class[i]:.5f}')
            except:
                print(f'Class {i} Accuracy: 0')

           
    test_loss /= len(test_loader)
    print ('correct:', correct)
    
    return test_loss, correct * 1.0 / (len(test_loader.dataset)) 



def main():
    parser = argparse.ArgumentParser(description = 'PyTorch Example')
    parser.add_argument('--batch_size', type = int, default = 48, metavar = 'N',
            help = 'input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type = int, default = 48, metavar = 'N',
            help = 'input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type = int, default = 150, metavar = 'N',
            help = 'number of epochs to train (default: 10)')
    parser.add_argument('--initial_lr', type = float, default = 0.0001, metavar = 'LR',
            help = 'learning rate (default: 0.01)')
    parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'M',
            help = 'SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action = 'store_true', default = True,
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
    parser.add_argument('--log_polar', type = bool, default = False, metavar = 'LP',
            help = 'include log polar transformation in training')
    parser.add_argument('--lp_out_shape', type = int, default = None, nargs = '*', metavar = 'N',
            help = 'output shape of log polar function')
    parser.add_argument('--salience', type = bool, default = False, metavar = 'SAL',
            help = 'use salience sampling to add augmentation')
    parser.add_argument('--salience_points', type = int, default = 1, metavar = 'N',
            help = 'number of points to sample')
    parser.add_argument('--training_aug', type = str, default = None,
            help = 'type of training augmentation')


    #required
    parser.add_argument('--dataset_path', type = str, required = True,
            help = 'path to dataset')
    parser.add_argument('--out_directory', type = str, required = True,
            help = 'directory to store experiments')
    parser.add_argument('--experiment_name', type = str, required = True,
            help = 'name for experiment')


    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}


#    outpath = '/checkpoint/mgahl/out/{}/{}'.format(args.out_directory, args.experiment_name)
    outpath = '/Users/marthagahl/Documents/Research-Gary/out_test/{}/{}'.format(args.out_directory, args.experiment_name)
    os.makedirs(outpath, exist_ok = True)

    num_classes = args.classes
    if args.lp_out_shape is not None:
        out_shape = tuple(args.lp_out_shape)
    else:
        out_shape = None

    ### Data loaders
#    train_dataset = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'train', 'identity', ['>30'])
#    val_dataset_1 = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'valid', 'identity', ['>30'])
#    val_dataset_2 = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'valid', 'identity', ['>30'])
#    test_dataset_1 = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'test', 'identity', ['>30'])
#    test_dataset_2 = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'test', 'identity', ['>30'])

    train_dataset = TransformedData("{}/train/".format(args.dataset_path), 180, 15, args.log_polar, out_shape, args.salience, args.salience_points, augmentation = args.training_aug, inversion = False)
    train_dataset_acc = TransformedData("{}/train/".format(args.dataset_path), 180, 0, args.log_polar, out_shape, args.salience, args.salience_points, augmentation = None, inversion = False)
    val_dataset_1 = TransformedData("{}/valid/".format(args.dataset_path), 180, 0, args.log_polar, out_shape, args.salience, args.salience_points, augmentation = None, inversion = False)
    val_dataset_2 = TransformedData("{}/valid/".format(args.dataset_path), 180, 0, args.log_polar, out_shape, args.salience, args.salience_points, augmentation = None, inversion = True)
    test_dataset_1 = TransformedData("{}/test/".format(args.dataset_path), 180, 0, args.log_polar, out_shape, args.salience, args.salience_points, augmentation = None, inversion = False)
    test_dataset_2 = TransformedData("{}/test/".format(args.dataset_path), 180, 0, args.log_polar, out_shape, args.salience, args.salience_points, augmentation = None, inversion = True)


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


    model = largeArchitecture(num_classes, out_shape)
#    model = nn.DataParallel(model).cuda()
    model = nn.DataParallel(model)


    train_losses = []
    train_accs = []
    val_upright_losses = []
    val_upright_accs = []
    val_inverted_losses = []
    val_inverted_accs = []

    low_val_upright_loss = np.inf

    optimizer =  optim.Adam(list(model.parameters()), lr = args.initial_lr, weight_decay = 1e-3)

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

        train_losses.append(float(str(train_loss)))
        train_accs.append(train_acc)

        val_upright_losses.append(float(str(val_upright_loss)))
        val_upright_accs.append(val_upright_acc)
        val_inverted_losses.append(float(str(val_inverted_loss)))
        val_inverted_accs.append(val_inverted_acc)

        json_data = {'train loss': str(train_loss), 
                     'train accuracy': str(train_acc), 
                     'val upright loss': str(val_upright_loss),
                     'val upright accuracy': str(val_upright_acc),
                     'val inverted loss': str(val_inverted_loss),
                     'val inverted accuracy': str(val_inverted_acc)
        }   
       
        with open(f'{outpath}/metrics_{epoch}.json','w') as f:
            json.dump(json_data, f)

    print()
    print(f'Training Losses: {train_losses}\n')
    print(f'Training Accuracies: {train_accs}\n')
    print(f'Upright Validation Losses: {val_upright_losses}\n')
    print(f'Upright Validation Accuracies: {val_upright_accs}\n')
    print(f'Inverted Validation Losses: {val_inverted_losses}\n')
    print(f'Inverted Validation Accuracies: {val_inverted_accs}\n')

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

