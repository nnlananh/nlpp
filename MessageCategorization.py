import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import warnings

with open("responses.json") as f:
    dataset = json.load(f)

NEGATIVE_WORDS = ["no", "don't", "without", "doesn't", "not", "isn't", "aren't", "wasn't", "weren't"]
TASKS = ["summarize", "paraphrase", "greetings", "questioning", "error"]
IGNORE_WORDS = ["?"]

class MessageCategorization:
    def __init__(self, message):
        self.words = []
        self.classes = set()
        self.message = message
        for intent in dataset:
            for pattern in intent["patterns"]:
                # Tokenize each word in the sentence.
                word = nltk.word_tokenize(pattern)
                # Add to words list.
                self.words.extend(word)
                # Add to classes list.
                self.classes.add(intent["tag"])


    def defining_task(self):

        self.task = ""