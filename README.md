# README

This repository is an attempt to reproduce the work 
described in the book chapter on "Voice presentation attack detection using
convolutional neural networks" (based on the original code at https://github.com/idiap/cnn-for-voice-antispoofing)

This version represents a rewrite of the original pytorch code which has been moved to the 
`original` subdirectory.  The aim is to clean up the code and clarify the assumptions
made.   The original version relies on matlab code for data preprocessing, here I have
implemented this in python using the SIDEKIT package. 
## Feature Extraction

The `features` directory contains the feature extraction code. It computes three sets of
features and could be extended to compute more or be more configurable. To compute features run:

```
python -m features.featureserver config/features.ini
```

`features.ini` just contains pointers to the data directories. It would be good to make it 
have more of the configuration for the feature extraction too.

## Model Training

The `models` directory contains the model training/testing code using pytorch.  This replicates 
the network architecture in the original code.  To train a model you need a configuration
file which points to the relevant training data and sets some learning parameters.

To train a test model use:

```
python -m models.train config/testconfig.ini
```
This will just use the data in `testdata` which has just 39 training files - the model won't be useful but it acts as a test of the software pipeline. 

Other configuration files are used to run different experiments.  The main difference being the
version of the training data they point to (different extracted features).

## Model Scoring

You can score a model using the `models.score` module:

```
python -m models.score testdata/models/dev-eer-8 
```

This will show the EER metric for the results.

See the notebook `notebooks/Experiment Results.ipynb` for an example of plotting ROC 
curves from experimental results.


