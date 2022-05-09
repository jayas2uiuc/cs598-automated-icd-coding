import re
import string
import itertools
from collections import Counter
import nltk
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import pkg_resources
from symspellpy import SymSpell, Verbosity
import dask.dataframe as dd
import multiprocessing
from dask.diagnostics import ProgressBar
import sys

'''Function to remove punctuations from notes'''
def removePunctuations(text):
    remove = string.punctuation.replace("'", "") 
    pattern = r"[{}]".format(remove) 
    return re.sub(pattern, " ", text) 

'''Function to replace digits with a special character d'''
def replaceDigitsWithd(text):
    return re.sub('\d', 'd', text)

'''Orchestrator function'''
def preprocessText(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = removePunctuations(text)
    text = replaceDigitsWithd(text)
    output = text.split(' ')
    output = list(filter(lambda a: a != '', output))
    return output

'''Function to get count of each word across records'''
def getWordCounts(data):
    combinedList = list(itertools.chain.from_iterable(data))
    countDict = dict(Counter(combinedList))
    return countDict

'''Function to find correct spelling of a word based on the count of other words'''
def findCorrectSpelling(inputWord, countDict, minFreq):  
    minDistance = 1e10
    correctSpelling = inputWord
    for word in countDict.keys():        
        if inputWord != word:
            distance = nltk.edit_distance(word, inputWord)
            if distance < minDistance:
                minDistance = distance
                correctSpelling = word
    return correctSpelling

'''Function to correct spellings of words'''
def getSpellingCorrections(countDict, minFreq = 5):
    corrections = {}
    words = countDict.keys()
    for word in words:
        if countDict[word] < minFreq:
            corrections[word] = findCorrectSpelling(word, countDict, minFreq)
    return corrections

def preprocess(inputdata):
    data = []

    for text in inputdata: 
        data.append(preprocessText(text))
         
    processedData = spellify(data, minFreq=5)
    return processedData

if __name__ == "__main__":

    dataset = pd.read_pickle(sys.argv[1])

    print(dataset.head())

    # Correct spellings
    print("Sym spell loading...")
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )

    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    tokenFreq = Counter(itertools.chain.from_iterable(dataset['PREPROCESSED_TEXT']))
    wrongSpellings = [] 
    print("Creating wrong spellings dictionary...")
    for token in tokenFreq:
        if tokenFreq[token] <= 5:
            wrongSpellings.append(token)
            
    word2idx = {k: i+1 for i, k in enumerate(tokenFreq.keys())}

    tqdm.pandas()

    def spellify(text):
        newRecord = []
        for word in text:
            if word in wrongSpellings:
                suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                if len(suggestions) > 0:
                    newRecord.append(suggestions[0].term)
            else:
                newRecord.append(word)
        return newRecord

    pbar = ProgressBar()
    pbar.register()


    print("Starting preprocessing...")
    with pbar:
        dataset['PAR_PREPROCESSED_TEXT'] = dd.from_pandas(dataset, npartitions=4*multiprocessing.cpu_count())\
        .map_partitions(lambda df: df.apply(lambda row: spellify(row.PREPROCESSED_TEXT), axis = 1))\
        .compute(scheduler="processes")

    print("Saving as pickle...")
    with open('./preprocessedMimic.pkl', 'wb') as handle:
        pickle.dump(dataset, handle)

