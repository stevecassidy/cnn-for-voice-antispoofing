
import os
import numpy as np
import torch


class ASVSpoofData(torch.utils.data.Dataset):
    def __init__(self, trials_fname, dirname):
        """
        Args:
            trials_fname: name of the file listing the trials (tab separated)
            dirname: directory name containing the data files
        """
        self.dirname = dirname
        self.labels = []
        self.fnames = []
        with open(trials_fname) as f:
            for line in f:
                parts = line.split()
                key = parts[0]
                val = parts[1]
                val = 0 if val == 'spoof' else 1
                basename, ext = os.path.splitext(key)
                self.labels.append(val)
                self.fnames.append(basename)
                 
        self.labels = torch.from_numpy(np.array(self.labels)).long()

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):        
        fname = os.path.join(self.dirname, self.fnames[idx]+'.npy')
        mat = np.load(fname)
        return (torch.from_numpy(mat).float(), self.labels[idx])

if __name__ == '__main__':
    d = ASVSpoofData('../data/ASVspoof2017/protocol_V2/ASVspoof2017_V2_dev.trl.txt', '../data/feat/narrow-wide/dev-files/')
    dl = torch.utils.data.DataLoader(d, batch_size=100)
    for x, l in dl:
        print("New batch")

