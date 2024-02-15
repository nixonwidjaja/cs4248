'''
    NUS CS4248 Assignment 1 - Objective 2 (Tokenization, Zipf's Law)

    Class Tokenizer for handling Objective 2

    Important: please strictly comply with the input/output formats for
               the method of tokenize_sentence, as we will call it in testing
'''
###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

# import matplotlib.pyplot as plt     # Requires matplotlib to create plots.
# import numpy as np    # Requires numpy to represent the numbers

# def draw_plot(r, f, imgname):
#     # Data for plotting
#     x = np.asarray(r)
#     y = np.asarray(f)

#     fig, ax = plt.subplots()
#     ax.plot(x, y)

#     ax.set(xlabel='Rank (log)', ylabel='Frequency (log)',
#         title='Word Frequency v.s. Rank (log)')
#     ax.grid()
#     fig.savefig(f"../plots/{imgname}")
#     plt.show()
from nltk.tokenize import word_tokenize
import re

class Tokenizer:

    def __init__(self, path, bpe=False, lowercase=True):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            self.text = f.read()
        
        self.bpe = bpe
        self.lowercase = lowercase

    def tokenize(self):
        ''' Returns/Saves a set of word tokens for the loaded textual file
        After calling this function, you should get a vocab set for the tokenizer,
        which you can utilize later in the method of tokenize_sentence.

        For the default setting, make sure you consider cases of:
            1) Words ending with punctuation (e.g., 'hiking.' ——> ['hiking', '.']);
            2) Numbers (e.g., '1/2', '12.5')
            3) Possessive case (e.g., "Elle's book" ——> ["elle's", "book"]). It's also fine if you follow 
               nltk's output for this case, we won't make a strict rule here.
            4) Title abbrevation - Mr., Mrs., Ms. ——> you can either remove the '.' before tokenization 
               or remain the token as a whole (e.g., 'Mr.' ——> 'mr.'). 
               You don't need to consider other special abbr like U.S., Ph.D., etc.

            For other corner cases (e.g. emoticons such as :D, units such as $, and other ill-formed English), you can check the output of 
            nltk.word_tokenize for reference. We won't punish you if your results are not the same as its outputs on other cases.

        For the bpe setting, 
            1) Tune the number of iterations so the vocab size will be close to the 
               default one's (approximately, the vocab size is about 13,000)
            2) During merge, for sub-sequences of the same frequency, break the tie 
               with left-to-right byte order precedence
        
        PS: This method is mandatory to implement to get a vocab 
            which you can utilize in tokenize_sentence
        '''
        # TODO Modify the code here
        nltk = set(word_tokenize(self.text.lower()))
        mine = set(self.tokenize_sentence(self.text))
        print(nltk.difference(mine))

    def basic_tokenize(self, sentence: str):
        if self.lowercase:
            sentence = sentence.lower()
        splits = sentence.split()
        tokens = []
        for s in splits:
            if s in ['mr.', 'ms.', 'mrs.', 'u.s.', 'ph.d.']:
                tokens.append(s)
                continue
            if s[0] == '$':
                tokens.append(s)
                continue
            if re.search('\W', s[0]): # punc start
                tokens.append(s[0])
                s = s[1:]
            if len(s) == 0:
                continue
            if s[-2:] in ["'s", '?"', '!"', '!”', '?”', ',”', '’s']:
                tokens.append(s[:-2])
                tokens.append(s[-2:])
                continue
            if re.search('\W', s[-1]): # punc end
                tokens.append(s[:-1])
                tokens.append(s[-1])
                continue
            else:
                tokens.append(s)
        return tokens
    
    def init_corpus(self, words):
        space_words = [' '.join(list(w)) for w in words]
        corpus = {}
        for w in space_words:
            if w not in corpus:
                corpus[w] = 0
            corpus[w] += 1
        return corpus
    
    def count_pairs(self, corpus):
        count = {}
        for s in corpus:
            chars = s.split()
            for i in range(len(chars) - 1):
                pair = chars[i] + chars[i+1]
                if pair not in count:
                    count[pair] = 0
                count[pair] += 1
        return list(count.items())
    
    def merge_pair(self, corpus, add_pair):
        new_corpus = {}
        for s in corpus:
            chars = s.split()
            new_s = ''
            i = 0
            while i < len(chars):
                if i == len(chars) - 1:
                    new_s += chars[i]
                    break
                pair = chars[i] + chars[i+1]
                if pair == add_pair:
                    new_s += pair + ' '
                    i += 1
                else:
                    new_s += chars[i] + ' '
                i += 1
            new_corpus[new_s] = corpus[s]
        return new_corpus
    
    def bpe_token_learner(self, sentence: str):
        if self.lowercase:
            sentence = sentence.lower()
        words = [i + '_' for i in sentence.split()]
        corpus = self.init_corpus(words)
        pairs = self.count_pairs(corpus)
        pair = max(pairs, key=lambda x: x[1])[0]
        corpus = self.merge_pair(corpus, pair)
    
    def tokenize_sentence(self, sentence: str):
        '''
        To verify your implementation, we will test this method by 
        input a sentence specified by us.  
        Please return the list of tokens as the result of tokenization.

        E.g. basic tokenizer (default setting)
        [In] sentence="I give 1/2 of the apple to my ten-year-old sister."
        [Out] return ['i', 'give', '1/2', 'of', 'the', 'apple', 'to', 'my', 'ten-year-old', 'sister', '.']
        
        Hint: For BPE, you may need to fix the vocab before tokenizing
              the input sentence
        
        PS: This method is mandatory to implement with the method signature as-is. 
        '''
        # TODO Modify the code here
        return self.basic_tokenize(sentence)
    
    def plot_word_frequency(self):
        '''
        Plot relative frequency versus rank of word to check
        Zipf's law
        You may want to use matplotlib and the function shown 
        above to create plots
        Relative frequency f = Number of times the word occurs /
                                Total number of word tokens
        Rank r = Index of the word according to word occurence list
        '''
        # TODO Modify the code here
        pass

    
if __name__ == '__main__':
    ##=== tokenizer initialization ===##
    basic_tokenizer = Tokenizer('../data/Pride_and_Prejudice.txt')
    bpe_tokenizer = Tokenizer('../data/Pride_and_Prejudice.txt', bpe=True)

    ##=== build the vocab ===##
    try:
        _ = basic_tokenizer.tokenize()  # for those which have a return value
    except:
        basic_tokenizer.tokenize()
    
    try:
        _ = bpe_tokenizer.tokenize()  # for those which have a return value
    except:
        bpe_tokenizer.tokenize()

    ##=== run on test cases ===##
    
    # you can edit the test_cases here to add your own test cases
    test_cases = ["""The Foundation's business office is located at 809 North 1500 West, 
        Salt Lake City, UT 84116, (801) 596-1887.""", 
        'I give 1/2 of the apple to my ten-year-old sister.']
    
    for case in test_cases:
        rst1 = basic_tokenizer.tokenize_sentence(case)
        rst2 = bpe_tokenizer.tokenize_sentence(case)

        ##= check the basic tokenizer =##
        # ['the', "foundation's", 'business', 'office', 'is', 'located', 'at', 
        # '809', 'north', '1500', 'west', ',', 'salt', 'lake', 'city', ',', 'ut', 
        # '84116', ',', '(', '801', ')', '596-1887', '.']
        # or
        # ['the', 'foundation', "'s", 'business', 'office', 'is', 'located', 'at', 
        # '809', 'north', '1500', 'west', ',', 'salt', 'lake', 'city', ',', 'ut', 
        # '84116', ',', '(', '801', ')', '596-1887', '.']
        print(rst1)

        ##= check the bpe tokenizer =##
        # ['the_', 'f', 'ou', 'n', 'd', 'a', 'ti', 'on', "'", 's_', 'bu', 
        # 's', 'in', 'es', 's_', 'o', 'f', 'f', 'i', 'c', 'e_', 'is_', 'l', 
        # 'o', 'c', 'at', 'ed_', 'at_', '8', '0', '9', '_', 'n', 'or', 'th_', 
        # '1', '5', '0', '0', '_', 'w', 'es', 't', ',_', 's', 'al', 't_', 'l', 
        # 'a', 'k', 'e_', 'c', 'it', 'y', ',_', 'u', 't_', '8', '4', '1', '1', 
        # '6', ',_', '(', '8', '0', '1', ')', '_', '5', '9', '6', '-', '1', '8', 
        # '8', '7', '._']
        print(rst2)