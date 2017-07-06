import nltk
# from utils import fnc_1_data
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pylab as py
# Libraries used
import time
import glob
import pandas as pd 
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords # Import the stop word list
import re
import requests
from bs4 import BeautifulSoup # Use bs4 to remove html tags or markup
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
#import itertools
#import numpy as np
#import matplotlib.pyplot as plt
#import itertools
#import numpy as np
#import matplotlib.pyplot as plt
#from keras import plot_confusion_matrix
# from utils import score_submission
import random
# import plot_confusion
#import tqdm as tqdm
#from tqdm import tnrange, tqdm_notebook
#from time import sleep
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import csv
from os import path
import random
from nltk.corpus import stopwords
import asyncio
import logging
import json
import os
from collections import OrderedDict

import watson_developer_cloud as wdc
#import websockets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


labels = ['High', 'Medium', 'Low']

def load_csv(fname):
    with open(fname) as csvfile:
        reader = csv.DictReader(csvfile)
        yield from reader
        
def train(datadir = './data'):
    train = list(load_csv(path.join(datadir, 'ibm_asr_dataset_cleaned.csv')))
    for article in train:
        y = article['Text']
    return train 
train = train('./data')
random.shuffle(train)


