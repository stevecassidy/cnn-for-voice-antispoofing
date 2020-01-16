
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by - Srikanth Madikeri (2017)

from __future__ import print_function
import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import shutil

from models.model import Net
from features.dataset import ASVSpoofData


def train(model, epoch, optimizer, train_loader, dev_loader):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.unsqueeze_(1)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    print("Total loss = %.6f" % (total_loss/len(train_loader.dataset)))

    test(model, dev_loader, 'dev-eer')
    


def test(model, test_loader, report_filename):
    model.eval()
    test_loss = 0
    correct = 0
    
    with open(report_filename, 'w') as f:
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.unsqueeze_(1)
            
            output = model(data)
            output_arr = output.data.numpy().reshape((-1, 2))
            tgt = target.data.numpy()
            for i in range(tgt.shape[0]):
                itgt = int(tgt[i])
                scr = - output_arr[i,0] + output_arr[i,1]
                if itgt == 0:
                    f.write('%f %s\n' % (scr, 'nontarget'))
                else:
                    f.write('%f %s\n' % (scr, 'target'))
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




if __name__=='__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='ConvNet reduction')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output-dir', type=str, default='./', metavar='N',
                        help='Output directory where models will be stored')
    parser.add_argument('--train', type=bool, default=True, 
                        help='Set/unset to toggle training')
    parser.add_argument('--eval', type=bool, default=True, 
                        help='Set/unset to toggle evaluation')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model_output_dir = args.output_dir + '/'
    try:
        if not os.path.isdir(model_output_dir):
            os.makedirs(model_output_dir)
    except Exception as e:
        print("Error creating model output directory", model_output_dir)
        print(e)

    print('Parameters: ', args.__dict__)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_dataset = ASVSpoofData('data/ASVspoof2017/protocol_V2/ASVspoof2017_V2_train.trn.txt', 
                                 'data/feat/narrow-wide/train-files/')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=True, 
                                            **kwargs)

    dev_dataset = ASVSpoofData('data/ASVspoof2017/protocol_V2/ASVspoof2017_V2_dev.trl.txt', 
                               'data/feat/narrow-wide/dev-files/')
    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=True, 
                                            **kwargs)

    model = Net()
    if args.cuda:
        model.cuda()

    optimizer =  optim.Adam(model.parameters(), lr=args.lr)

    if args.train:
        for epoch in range(1, args.epochs):
            train(model, epoch, optimizer, train_loader, dev_loader)
            torch.save(model, model_output_dir + 'iter' + str(epoch) + '.mdl')
            shutil.copy('dev-eer', model_output_dir + '/dev-eer-' + str(epoch))

        torch.save(model, model_output_dir + 'final.mdl')

    if args.eval:
        test()
