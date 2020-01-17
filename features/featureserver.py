"""
Code to create feature servers in Sidekit

"""

import sidekit
import os
import glob
import math
import numpy
import logging
import configparser


def extract_features(config, dirname, outdir):
    """
    Extract features from a set of data in DATA_DIR/dirname (*.wav)
    and write features into FEAT_DIR/dirname for future use
    """
    # TODO: more of these settings should be derived from the config file

    dd = os.path.join(config['DATA_DIR'], dirname)
    fd768 = os.path.join(config['FEAT_DIR'], "wideband-768", outdir)
    fd_concat = os.path.join(config['FEAT_DIR'], "narrow-wide", outdir)

    if not os.path.exists(fd768):
        os.makedirs(fd768)

    if not os.path.exists(fd_concat):
        os.makedirs(fd_concat)

    print("Extracting features for data in {} to {} and {}".format(dd, fd768, fd_concat))

    fs768 = make_feature_server(dd, 768)
    fs641 = make_feature_server(dd, 641)
    fs127 = make_feature_server(dd, 127)

    for wavfilename in glob.glob(os.path.join(dd, '*.wav')):
        
        basename, _ext = os.path.splitext(os.path.basename(wavfilename))
        print("Processing {}".format(basename))

        feat768, _label = fs768.load(basename)
        feat641, _label = fs641.load(basename)
        feat127, _label = fs127.load(basename)

        outname = os.path.join(fd768, basename)
        numpy.save(outname, normalize(ensure_400(feat768.transpose())))

        # combine the 641 and 127 features, 127 will be a bit longer so chop it down
        width = feat641.shape[0]
        feat_combined = numpy.concatenate((feat641, feat127[:width,:]), axis=1)
        outname = os.path.join(fd_concat, basename)
        numpy.save(outname, normalize(ensure_400(feat_combined.transpose())))

def ensure_400(mat):
    """Stretch (by duplicating) or shrink this array to ensure that it is 400x768"""

    size = mat.shape[1]
    # make sure that mat is at least 400 wide, repeat if not
    if size < 400:
        rr = int(math.ceil(400.0/size)) 
        mat = numpy.tile(mat,rr)

    # return the first 400 cols
    return mat[:,0:400]


def normalize(mat):
    """Normalise data"""

    nFeatures = 768

    mat = (mat.reshape(nFeatures,-1) - numpy.mean(mat.reshape(nFeatures,-1),axis=1,keepdims=True))
    mat = numpy.divide(mat.reshape(nFeatures,-1),numpy.std(mat.reshape(nFeatures,-1),axis=1,keepdims=True))

    return mat


def make_feature_server(dirname, frame_size):
    """Return a Sidekit FeatureServer instance for this
    experiement
    """

    sampling_frequency = 16000
    # window size must be twice the frame size to give the right number of FFT points but since
    # we can't zero pad, we'll be taking in more of the signal in each frame
    window_size =  (2* frame_size+1) / sampling_frequency
    shift = 0.008

    # make a feature server to compute features over our audio files
    extractor = sidekit.FeaturesExtractor(audio_filename_structure=dirname + "/{}.wav",
                                          sampling_frequency=sampling_frequency,
                                          lower_frequency=0,
                                          higher_frequency=sampling_frequency/2,
                                          filter_bank="lin",
                                          filter_bank_size=frame_size,
                                          window_size=window_size,
                                          shift=shift,
                                          ceps_number=20, 
                                          pre_emphasis=0.97,
                                          save_param=["fb"],
                                          keep_all_features=False)

    server = sidekit.FeaturesServer(features_extractor=extractor,
                                    sources=None,
                                    dataset_list=["fb"],
                                    keep_all_features=True)

    return server


if __name__ == '__main__':
    
    import sys

    configfile = sys.argv[1]

    CONFIG = configparser.ConfigParser()
    CONFIG.read(configfile)

    extract_features(CONFIG['default'], "ASVspoof2017_V2_train", "train-files")
    extract_features(CONFIG['default'], "ASVspoof2017_V2_dev", "dev-files")
    extract_features(CONFIG['default'], "ASVspoof2017_V2_eval", "eval-files")

