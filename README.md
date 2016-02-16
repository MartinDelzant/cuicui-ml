# cuicui-ml
Bird species classification using their songs

Check out the ipython notebook ;)

## How it works

The data should be downloaded from : 

 - Train data : http://sis.univ-tln.fr/~glotin/SABIOD/BIRD50/AmazonBird50_training_input.tar.gz
 - Test data : http://sis.univ-tln.fr/~glotin/SABIOD/BIRD50/AmazonBird50_testing_input.tar.gz

And placed in subdirectories : AmazonBird50_training_input and AmazonBird50_testing_input respectively

## Dependencies

Numpy, Scipy, Sklearn, python_speech_features :
```
git clone https://github.com/jameslyons/python_speech_features
cd python_speech_features
python setup.py install
```

openSMILE : https://web.stanford.edu/class/cs224s/hw/openSMILE_manual.pdf

## Data

The data comes from ... (TO be completed)

## Prediction 

Détails des submits :

ypred2.csv : Welch features (1000:10000:500 -> 2 by 2) + RDF 800 estim, max_feat 5.
7-Cross-val score ~31%
Leaderboard : 32%

