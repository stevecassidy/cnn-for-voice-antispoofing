
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by - Srikanth Madikeri (2017)

from __future__ import print_function
import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import shutil
import configparser

from models.model import Net
from features.dataset import ASVSpoofData
from models.score import eer

def make_dataloader(speaker_list_file, dirname, batch_size, kwargs={}):
    """Make and return a data loader object"""

    dataset = ASVSpoofData(speaker_list_file, dirname)

    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size, 
                                        shuffle=True, **kwargs)

    return loader


def measure_size(model):
    """Measure the number of parameters in the model"""

    print('Conv1:', model.conv1.weight.size())
    print('Conv2:', model.conv2.weight.size())
    print('Conv3:', model.conv3.weight.size())
    print('Fc1:', model.fc1.weight.size())
    print('Fc2:', model.fc2.weight.size())
    return sum(p.numel() for p in model.parameters())


def train(epochs, train_loader, dev_loader, lr, seed, log_interval, output_dir):
    """Train the model. Store snapshot models in the output_dir alongside
    evaluations on the dev set after each epoch
    """

    model = Net()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    measure_size(model)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: ", device)

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    model.to(device)
    
    for epoch in range(1, epochs):

        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.to(device), target.to(device)
            data = data.unsqueeze_(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

        print("Total loss = %.6f" % (total_loss/len(train_loader.dataset)))

        test(model, dev_loader, os.path.join(output_dir, 'dev-eer-' + str(epoch)))         

        torch.save(model, os.path.join(output_dir, 'iter' + str(epoch) + '.mdl'))


def test(model, test_loader, report_filename):
    """Test the model on a provided dataset
    Save test output to a `report_filename`.
    Print a summary of performance to stdout.
    """
    
    model.eval()
    test_loss = 0
    correct = 0
    scores = []
    targets = []
    
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
   

    with open(report_filename, 'w') as f:
        for data, target in test_loader:
            if use_cuda:
                data, target = data.to(device), target.to(device)
            data = data.unsqueeze_(1)
            
            output = model(data)
            if use_cuda:
                op = output.cpu()
                tgt = target.cpu()
            else:
                op = output
                tgt = target
            output_arr = op.data.numpy().reshape((-1, 2))
            tgt = tgt.data.numpy()
            for i in range(tgt.shape[0]):
                itgt = int(tgt[i])
                scr = - output_arr[i,0] + output_arr[i,1]
                t = "nontarget" if itgt == 0 else "target"
                f.write('%f %s\n' % (scr, t))
                scores.append(scr)
                targets.append(t)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_eer = eer(targets, scores)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), EER: {}%\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        test_eer*100))


if __name__=='__main__':


    import sys

    configfile = sys.argv[1]

    CONFIG = configparser.ConfigParser()
    CONFIG.read(configfile)
    config = CONFIG['default']

    model_output_dir = config['MODEL_DIR']
    data_dir = config['DATA_DIR']
    feat_dir = config['FEAT_DIR']

    try:
        if not os.path.isdir(model_output_dir):
            os.makedirs(model_output_dir)
    except Exception as e:
        print("Error creating model output directory", model_output_dir)
        print(e)

    use_cuda = config.getboolean('CUDA') and torch.cuda.is_available()

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = make_dataloader(os.path.join(data_dir, 'protocol_V2/ASVspoof2017_V2_train.trn.txt'), 
                                   os.path.join(feat_dir, 'train-files/'),
                                   config.getint('BATCH_SIZE'), kwargs)

    dev_loader = make_dataloader(os.path.join(data_dir, 'protocol_V2/ASVspoof2017_V2_dev.trl.txt'), 
                                 os.path.join(feat_dir, 'dev-files/'),
                                 config.getint('BATCH_SIZE'), kwargs)

    model = train(
                    epochs=config.getint('EPOCHS'), 
                    train_loader=train_loader, 
                    dev_loader=dev_loader,
                    lr=config.getfloat('LEARNING_RATE'), 
                    seed=config.getfloat('seed', 1),
                    log_interval=config.getint('LOG_INTERVAL', 10),
                    output_dir=model_output_dir
                )

    torch.save(model, model_output_dir + 'final.mdl')


