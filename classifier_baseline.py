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

import pathlib
from PIL import Image

from dataset_baseline import Collater, CelebADataset


dataset_name = 'celeba'
experiment_type = 'baseline'
num_classes = 2
upright_rotations = [0]
inverted_rotations = [180]
num_crops = 10

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
    return average_loss


def test(args, model, device, test_loader, num_crops, num_rotations):
    model.eval()
    test_loss = 0
    correct = 0
    
    num_class = {}
    class_correct = {}
    top_17 = [338, 435, 573, 681, 769, 779, 785, 855, 906, 1052, 1088, 1122, 1605, 2120, 2173, 2198, 2265]

    with torch.no_grad():
        for data, target in test_loader:
            data = data.type(torch.cuda.FloatTensor)
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
    
    return test_loss, correct * 1.0 / (len(test_loader.dataset)*num_crops*num_rotations) 




def main():
    parser = argparse.ArgumentParser(description = 'PyTorch Example')
    parser.add_argument('--batch-size', type = int, default = 4, metavar = 'N',
            help = 'input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type = int, default = 4, metavar = 'N',
            help = 'input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type = int, default = 100, metavar = 'N',
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
    parser.add_argument('--log_polar', type = bool, default = False, metavar = 'LP',
            help = 'include log polar transformation in training')
    parser.add_argument('--salience', type = bool, default = False, metavar = 'SAL',
            help = 'use salience sampling to add augmentation')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}


    outpath = '/checkpoint/mgahl/out/{}/{}'.format(dataset_name,experiment_type)
    os.makedirs(outpath, exist_ok = True)


    ### Data loaders
    train_dataset = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'train', 'identity', ['>30'])
    val_dataset_1 = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'valid', 'identity', ['>30'])
    val_dataset_2 = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'valid', 'identity', ['>30'])
    test_dataset_1 = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'test', 'identity', ['>30'])
    test_dataset_2 = CelebADataset('/datasets01/CelebA/CelebA/072017/img_align_celeba', 'test', 'identity', ['>30'])

#    train_dataset = datasets.ImageFolder("/checkpoint/mgahl/shape_dataset/train/")
#    val_dataset_1 = datasets.ImageFolder("/checkpoint/mgahl/shape_dataset/valid/")
#    val_dataset_2 = datasets.ImageFolder("/checkpoint/mgahl/shape_dataset/valid/")
#    test_dataset_1 = datasets.ImageFolder("/checkpoint/mgahl/shape_dataset/test/")
#    test_dataset_2 = datasets.ImageFolder("/checkpoint/mgahl/shape_dataset/test/")

    train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size = args.batch_size, 
            shuffle = True, 
            collate_fn = Collater(args.log_polar, 150, upright_rotations), 
            **kwargs)
    val_loader_1 = torch.utils.data.DataLoader(
            val_dataset_1, 
            batch_size = args.test_batch_size, 
            shuffle = True, 
            collate_fn = Collater(args.log_polar, 150, upright_rotations), 
            **kwargs)
    val_loader_2 = torch.utils.data.DataLoader(
            val_dataset_2, 
            batch_size = args.test_batch_size, 
            shuffle = True, 
            collate_fn = Collater(args.log_polar, 150, inverted_rotations), 
            **kwargs)
    test_loader_1 = torch.utils.data.DataLoader(
            test_dataset_1, 
            batch_size = args.test_batch_size, 
            shuffle = True, 
            collate_fn = Collater(args.log_polar, 150, upright_rotations), 
            **kwargs)
    test_loader_2 = torch.utils.data.DataLoader(
            test_dataset_2, 
            batch_size = args.test_batch_size, 
            shuffle = True, 
            collate_fn = Collater(args.log_polar, 150, inverted_rotations), 
            **kwargs)



    class Flatten(nn.Module):
        def forward(self,input):
#            print (input.shape)
            return input.view(input.size(0), -1)


    model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 3, bias = False ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
#            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = False),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            Flatten(),
            nn.Linear(46208, 40),
#            nn.Linear(10368, 40),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(40, num_classes),
            )


    model = nn.DataParallel(model).cuda()

    def get_optimizer(initial_lr, epoch):
        ratio = 0.1  # set 1/10 as the ratio ??? 
        lr = initial_lr * math.pow(ratio, epoch/40)
        print ("The current Learning Rate is {}.".format(lr))
        return optim.Adam(list(model.parameters()), lr = lr, weight_decay = 1e-3)

    train_losses = []
    train_accs = []
    val_upright_losses = []
    val_upright_accs = []
    val_inverted_losses = []
    val_inverted_accs = []

    low_val_upright_loss = np.inf

    for epoch in range(0, args.epochs):
        optimizer = get_optimizer(args.initial_lr, epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        
        print("\nEvaluating on training set...")
        train_loss, train_acc = test(args, model, device, train_loader, num_crops, len(upright_rotations))
        print("\nEvaluating on upright validation set...")
        val_upright_loss, val_upright_acc = test(args, model, device, val_loader_1, num_crops, len(upright_rotations))
        print("\nEvaluating on inverted validation set...")
        val_inverted_loss, val_inverted_acc = test(args, model, device, val_loader_2, num_crops, len(inverted_rotations))


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
    test_upright_loss, test_upright_acc = test(args, model, device, test_loader_1, num_crops, len(upright_rotations))
    print("\nEvaluating on inverted test set...")
    test_inverted_loss, test_inverted_acc = test(args, model, device, test_loader_2, num_crops, len(inverted_rotations))

    print(f'\nUpright Test Set Loss: {test_upright_loss}')
    print(f'Upright Test Set Accuracy: {test_upright_acc}')
    print(f'\nInverted Test Set Loss: {test_inverted_loss}')
    print(f'Inverted Test Set Accuracy: {test_inverted_acc}')


if __name__ == '__main__':
    main()

