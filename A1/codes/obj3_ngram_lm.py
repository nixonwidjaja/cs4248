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
        self.LM = {}
        self.special_tokens = {'bos': '~', 'eos': '<EOS>'}
        text = self.read_file(path)
        tokens = self.tokenize(text)
        if self.n == 1:
            self.generate_unigram_LM(tokens)
            self.words = list(self.LM.keys())
            self.weights = list(self.LM.values())
        else:
            self.generate_bigram_LM(tokens)
            self.words = {}
            self.weights = {}
            for a in self.LM:
                self.words[a] = list(self.LM[a].keys())
                self.weights[a] = list(self.LM[a].values())

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
                self.LM[t] = self.k
            self.LM[t] += 1
        self.total = 0
        for i in self.LM:
            self.total += self.LM[i]
            
    def generate_smoothing_all_words(self, token_set: set):
        d = {}
        for t in token_set:
            d[t] = self.k
        return d

    def generate_bigram_LM(self, tokens):
        # TODO Write your code here
        token_set = set(tokens)
        for i in range(len(tokens) - 1):
            a = tokens[i]
            b = tokens[i + 1]
            if a not in self.LM:
                self.LM[a] = self.generate_smoothing_all_words(token_set)
            self.LM[a][b] += 1
        self.total = {}
        for a in self.LM:
            self.total[a] = 0
            for b in self.LM[a]:
                self.total[a] += self.LM[a][b]

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
            randomList = random.choices(self.words, weights=self.weights)
        else:
            tokens = [self.special_tokens['bos']] + word_tokenize(text)
            prev = tokens[-1]
            randomList = random.choices(self.words[prev], weights=self.weights[prev])
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
                    if t in self.LM:
                        pp -= math.log(self.LM[t] / self.total)
            else:
                for i in range(len(tokens) - 1):
                    a = tokens[i]
                    b = tokens[i + 1]
                    if a and b:
                        pp -= math.log(self.LM[a][b] / self.total[a])
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
    print(f"{end - start} seconds")