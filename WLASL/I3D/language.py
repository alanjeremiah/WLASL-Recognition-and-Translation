# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 09:37:10 2022

@author: 24412
"""

from itertools import chain
import math
import os


from dotenv import load_dotenv
import nltk
import random
import attr

from collections import Counter

load_dotenv("posts/nlp/.env", override=True)
@attr.s(auto_attribs=True)
class NGrams:
    """The N-Gram Language Model

    Args:
     data: the training data
     n: the size of the n-grams
     start_token: string to represent the start of a sentence
     end_token: string to represent the end of a sentence
    """
    data: list
    n: int
    start_token: str="<s>"
    end_token: str="<e>"
    _start_tokens: list=None
    _end_tokens: list=None
    _sentences: list=None
    _n_grams: list=None
    _counts: dict=None

    @property
    def start_tokens(self) -> list:
        """List of 'n' start tokens"""
        if self._start_tokens is None:
            self._start_tokens = [self.start_token] * self.n
        return self._start_tokens
    
    @property
    def end_tokens(self) -> list:
        """List of 1 end-tokens"""
        if self._end_tokens is None:
            self._end_tokens = [self.end_token]
        return self._end_tokens
    
    @property
    def sentences(self) -> list:
        """The data augmented with tags and converted to tuples"""
        if self._sentences is None:
            self._sentences = [tuple(self.start_tokens + sentence + self.end_tokens)
                              for sentence in self.data]
        return self._sentences
    
    @property
    def n_grams(self) -> list:
        """The n-grams from the data

        Warning:
        this flattens the n-grams so there isn't any sentence structure
        """
        if self._n_grams is None:
            self._n_grams = chain.from_iterable([
                [sentence[cut: cut + self.n] for cut in range(0, len(sentence) - (self.n - 1))]
                for sentence in self.sentences
            ])
        return self._n_grams
   
    @property
    def counts(self) -> Counter:
        """A count of all n-grams in the data

        Returns:
          A dictionary that maps a tuple of n-words to its frequency
        """
        if self._counts is None:        
            self._counts = Counter(self.n_grams)
        return self._counts

@attr.s(auto_attribs=True)
class Tokenizer:
    """Tokenizes string sentences

    Args:
     source: string data to tokenize
     end_of_sentence: what to split sentences on

    """
    source: str
    end_of_sentence: str="\n"
    _sentences: list=None
    _tokenized: list=None
    _training_data: list=None

    
    @property
    def sentences(self) -> list:
      """The data split into sentences"""
      if self._sentences is None:
          self._sentences = self.source.split(self.end_of_sentence)
          self._sentences = (sentence.strip() for sentence in self._sentences)
          self._sentences = [sentence for sentence in self._sentences if sentence]
      return self._sentences

    @property
    def tokenized(self) -> list:
        """List of tokenized sentence"""
        if self._tokenized is None:
            self._tokenized = [nltk.word_tokenize(sentence.lower())
                              for sentence in self.sentences]
        return self._tokenized
    
    
@attr.s(auto_attribs=True)
class TrainTestSplit:
    """splits up the training and testing sets

    Args:
     data: list of data to split
     training_fraction: how much to put in the training set
     seed: something to seed the random call
    """
    data: list
    training_fraction: float=0.8
    seed: int=87
    _shuffled: list=None
    _training: list=None
    _testing: list=None
    _split: int=None

    @property
    def shuffled(self) -> list:
        """The data shuffled"""
        if self._shuffled is None:
            random.seed(self.seed)
            self._shuffled = random.sample(self.data, k=len(self.data))
        return self._shuffled

    @property
    def split(self) -> int:
        """The slice value for training and testing"""
        if self._split is None:
            self._split = int(len(self.data) * self.training_fraction)
        return self._split

    @property
    def training(self) -> list:
        """The Training Portion of the Set"""
        if self._training is None:
            self._training = self.shuffled[0:self.split]
        return self._training

    @property
    def testing(self) -> list:
        """The testing data"""
        if self._testing is None:
            self._testing = self.shuffled[self.split:]
        return self._testing
def estimate_probability(word: str,
                         previous_n_gram: tuple, 
                         n_gram_counts: dict,
                         n_plus1_gram_counts: dict,
                         vocabulary_size: int,
                         k: float=1.0) -> float:
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing

    Args:
       word: next word
       previous_n_gram: A sequence of words of length n
       n_gram_counts: Dictionary of counts of n-grams
       n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
       vocabulary_size: number of words in the vocabulary
       k: positive constant, smoothing parameter

    Returns:
       A probability
    """
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)

    n_plus1_gram = previous_n_gram + (word,)  
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)       
    return (n_plus1_gram_count + k)/(previous_n_gram_count + k * vocabulary_size)


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing

    Args:
       previous_n_gram: A sequence of words of length n
       n_gram_counts: Dictionary of counts of (n+1)-grams
       n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
       vocabulary: List of words
       k: positive constant, smoothing parameter

    Returns:
       A dictionary mapping from next words to the probability.
    """

    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, 
                                           n_gram_counts, n_plus1_gram_counts, 
                                           vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities

def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    """
    Get suggestion for the next word

    Args:
       previous_tokens: The sentence you input where each token is a word. Must have length > n 
       n_gram_counts: Dictionary of counts of (n+1)-grams
       n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
       vocabulary: List of words
       k: positive constant, smoothing parameter
       start_with: If not None, specifies the first few letters of the next word

    Returns:
       A tuple of 
         - string of the most likely next word
         - corresponding probability
    """

    # length of previous words
    n = len(list(n_gram_counts.keys())[0]) 

    # From the words that the user already typed
    # get the most recent 'n' words as the previous n-gram
    previous_n_gram = previous_tokens[-n:]

    # Estimate the probabilities that each word in the vocabulary
    # is the next word,
    # given the previous n-gram, the dictionary of n-gram counts,
    # the dictionary of n plus 1 gram counts, and the smoothing constant
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)

    # Initialize suggested word to None
    # This will be set to the word with highest probability
    suggestion = None

    # Initialize the highest word probability to 0
    # this will be set to the highest probability 
    # of all words to be suggested
    max_prob = 0

    ### START CODE HERE (Replace instances of 'None' with your code) ###

    # For each word and its probability in the probabilities dictionary:
    for word, prob in probabilities.items(): # complete this line

        # If the optional start_with string is set
        if start_with is not None: # complete this line

            # Check if the beginning of word does not match with the letters in 'start_with'
            if not word.startswith(start_with): # complete this line

                # if they don't match, skip this word (move onto the next word)
                continue # complete this line

        # Check if this word's probability
        # is greater than the current maximum probability
        if prob > max_prob: # complete this line

            # If so, save this word as the best suggestion (so far)
            suggestion = word

            # Save the new maximum probability
            max_prob = prob

    ### END CODE HERE

    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    #for i in range(model_counts-1):
    n_gram_counts = n_gram_counts_list[0]
    n_plus1_gram_counts = n_gram_counts_list[1]

    suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                n_plus1_gram_counts, vocabulary,
                                k=k, start_with=start_with)
    return " " + str(suggestion[0])


