'''
    NUS CS4248 Assignment 1 - Objective 3 (n-gram Language Model)

    Class NgramLM for handling Objective 3

    Important: please strictly comply with the input/output formats for
               the methods of generate_word & generate_text & get_perplexity, 
               as we will call them during testing

    Sentences for Task 3:
    1) "They just entered a beautiful walk by"
    2) "The rabbit hopped onto a beautiful walk by the garden."
    3) "They had just spotted a snake entering"
'''

###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import random, math
import time
from nltk.tokenize import word_tokenize, sent_tokenize


class NgramLM(object):

    def __init__(self, path: str, n: int, k: float):
        '''This method is mandatory to implement with the method signature as-is.

            Initialize your n-gram LM class

            Parameters:
                n (int) : order of the n-gram model
                k (float) : smoothing hyperparameter

            Suggested function dependencies:
                read_file -> init_corpus |-> get_ngrams_from_seqs -> add_padding_to_seq
                                         |-> get_vocab_from_tokens

                generate_text -> generate_word -> get_next_word_probability

                get_perplexity |-> get_ngrams_from_seqs
                               |-> get_next_word_probability

        '''
        # Initialise other variables as necessary
        # TODO Write your code here
        self.n = n
        self.k = k

        # Fields below are optional but recommended; you may replace as you like
        self.LM = {} # store count
        self.special_tokens = {'bos': '~', 'eos': '<EOS>'}
        text = self.read_file(path)
        tokens = self.tokenize(text)
        self.V = len(set(tokens))
        if self.n == 1:
            self.generate_unigram_LM(tokens)
            self.words = list(self.LM.keys())
        else:
            self.generate_bigram_LM(tokens)
            self.words = {}
            for a in self.LM:
                self.words[a] = list(self.LM[a].keys())

    def read_file(self, path: str):
        # TODO Write your code here
        with open(path, 'r') as f:
            text = f.read()
        return text.lower()

    def tokenize(self, text: str):
        # TODO Write your code here
        sentences = sent_tokenize(text)
        tokens = []
        for sen in sentences:
            tokens.append(self.special_tokens['bos'])
            tokens.extend(word_tokenize(sen))
            tokens.append(self.special_tokens['eos'])
        return tokens

    def generate_unigram_LM(self, tokens):
        # TODO Write your code here
        for t in tokens:
            if t not in self.LM:
                self.LM[t] = 0
            self.LM[t] += 1
        self.count_vocab = 0
        for i in self.LM:
            self.count_vocab += self.LM[i]
            
    def init_count(self, token_set: set):
        d = {}
        for t in token_set:
            d[t] = 0
        return d

    def generate_bigram_LM(self, tokens):
        # TODO Write your code here
        token_set = set(tokens)
        for i in range(len(tokens) - 1):
            a = tokens[i]
            b = tokens[i + 1]
            if a not in self.LM:
                self.LM[a] = self.init_count(token_set)
            self.LM[a][b] += 1
        self.count_vocab = {}
        for a in self.LM:
            self.count_vocab[a] = 0
            for b in self.LM[a]:
                self.count_vocab[a] += self.LM[a][b]

    def count_probability(self, a, b=None):
        if self.n == 1:
            return (self.LM.get(a, 0) + self.k) / (self.count_vocab + self.k * self.V)
        else:
            if a in self.LM:
                return (self.LM[a].get(b, 0) + self.k) / (self.count_vocab[a] + self.k * self.V)
            return (self.LM.get(a, 0) + self.k) / (self.count_vocab.get(a, 0) + self.k * self.V)

    def generate_word(self, text: str):
        '''
        Generates a random word based on the specified text and the ngrams learned
        by the model

        PS: This method is mandatory to implement with the method signature as-is.
            We only test one sentence at a time, so you may not need to split 
            the text into sentences here.

        [In] string (a full sentence or half of a sentence)
        [Out] string (a word)
        '''
        # TODO Write your code here
        text = text.lower()
        if self.n == 1:
            probs = [self.count_probability(w) for w in self.words]
            randomList = random.choices(self.words, weights=probs)
        else:
            tokens = [self.special_tokens['bos']] + word_tokenize(text)
            prev = tokens[-1]
            prev = prev if prev in self.words else self.special_tokens['bos']
            probs = [self.count_probability(prev, w) for w in self.words[prev]]
            randomList = random.choices(self.words[prev], weights=probs)
        return randomList[0]

    def generate_text(self, length: int):
        ''' Generate text of a specified length based on the learned ngram model 
        
        [In] int (length: number of tokens)
        [Out] string (text)

        PS: This method is mandatory to implement with the method signature as-is. 
            The length here is a reasonable int number, (e.g., 3~20)
        '''
        # TODO Write your code here
        text = []
        sentence = ''
        for _ in range(length):
            text.append(self.generate_word(sentence))
            sentence = ' '.join(text)
        return sentence

    def get_perplexity(self, text: str):
        '''
        Returns the perplexity of texts based on learned ngram model. 
        Note that text may be a concatenation of multiple sequences.
        
        [In] string (a short text)
        [Out] float (perplexity) 

        PS: This method is mandatory to implement with the method signature as-is. 
            The output is the perplexity, not the log form you use to avoid 
            numerical underflow in calculation.

        Hint: To avoid numerical underflow, add logs instead of multiplying probabilities.
              Also handle the case when the LM assigns zero probabilities.
        '''
        # TODO Write your code here
        text = text.lower()
        pp = 0
        sentences = sent_tokenize(text)
        N = 0
        for sen in sentences:
            tokens = [self.special_tokens['bos']] + word_tokenize(sen) + [self.special_tokens['eos']]
            N += len(tokens)
            if self.n == 1:
                for t in tokens:
                    pp -= math.log(self.count_probability(t))
            else:
                for i in range(len(tokens) - 1):
                    a = tokens[i]
                    b = tokens[i + 1]
                    pp -= math.log(self.count_probability(a, b))
        pp /= N
        return math.exp(pp)


if __name__ == '__main__':
    print('''[Alert] Time your code and make sure it finishes within 1 minute!''')

    start = time.time()
    LM = NgramLM('../data/Pride_and_Prejudice.txt', n=2, k=1.0)

    test_cases = ["The rabbit hopped onto a beautiful walk by the garden.", 
        "They just entered a beautiful walk by", 
        "They had just spotted a snake entering"]

    for case in test_cases:
        word = LM.generate_word(case)
        ppl = LM.get_perplexity(case)
        print(f'input text: {case}\nnext word: {word}\nppl: {ppl}')
    
    _len = 7
    text = LM.generate_text(length=_len)
    print(f'\npredicted text of length {_len}: {text}')
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")