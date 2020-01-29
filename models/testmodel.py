import torch

from models.train import test, make_dataloader
from models.model import Net


if __name__=='__main__':

    import sys, os

    model_file = sys.argv[1]
    report_filename = sys.argv[2]

    loader = make_dataloader(os.path.join('s3data', 'protocol_V2/ASVspoof2017_V2_dev.trl.txt'), 
                             os.path.join('s3data/wideband-768', 'dev-files/'),
                             10)

    # load a saved model
    device = torch.device('cpu')
    model = Net()
    #model.load_state_dict()
    model = torch.load(model_file, map_location=device)

    print("model", model_file, model)
    # test it

    test(model, loader, report_filename)