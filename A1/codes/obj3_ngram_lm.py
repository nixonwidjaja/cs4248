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

# import random, math
import collections


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
        self.ngram_dict = collections.defaultdict()
        self.special_tokens = {'bos': '~', 'eos': '<EOS>'}

    def read_file(self, path: str):
        ''' Reads text from file path and initiate n-gram corpus.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of __init__ 
        '''
        # TODO Write your code here
        pass

    def init_corpus(self, text: str):
        ''' Initiates n-gram corpus based on loaded text
        
        PS: Change the function signature as you like. 
            This method is only a suggested method,
            which you may call in the method of read_file 
        '''
        # TODO Write your code here
        pass

    def get_vocab_from_tokens(self, tokens):
        ''' Returns the vocabulary (e.g. {word: count}) from a list of tokens

        Hint: to get the vocabulary, you need to first tokenize the corpus.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of init_corpus.
        '''
        # TODO Write your code here
        pass

    def get_ngrams_from_seqs(self, sentences):
        ''' Returns ngrams of the text as list of pairs - [(sequence context, word)] 
            where sequence context is the ngram and word is its last word

        Hint: to get the ngrams, you may need to first get split sentences from corpus,
            and add paddings to them.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of init_corpus 
        '''
        # TODO Write your code here
        pass

    def add_padding_to_seq(self, sentence: str):
        '''  Adds paddings to a sentence.
        The goal of the method is to pad start token(s) to input sentence,
        so that we can get token '~ I' from a sentence 'I like NUS.' as in the bigram case.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of get_ngrams_from_seqs 
        '''
        # TODO Write your code here
        # Use '~' as your padding symbol
        pass

    def get_next_word_probability(self, text: str, word: str):
        ''' Returns probability of a word occurring after specified text, 
        based on learned ngrams.

        PS: Change the function signature as you like. 
            This method is a suggested method to implement,
            which you may call in the method of generate_word         
        '''
        # TODO Write your code here
        pass

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
        pass

    def generate_text(self, length: int):
        ''' Generate text of a specified length based on the learned ngram model 
        
        [In] int (length: number of tokens)
        [Out] string (text)

        PS: This method is mandatory to implement with the method signature as-is. 
            The length here is a reasonable int number, (e.g., 3~20)
        '''
        # TODO Write your code here
        pass

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
        pass


if __name__ == '__main__':
    print('''[Alert] Time your code and make sure it finishes within 1 minute!''')

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