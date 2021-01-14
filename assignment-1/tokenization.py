import csv
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, WhitespaceTokenizer
import spacy

datafile = 'dataset/IRAhandle_tweets_1.csv'

def get_tweets(filename):
    tweets = []
    with open(datafile, 'r') as csvfile:
        data = csv.DictReader(csvfile)
        for row in data:
            tweets.append(row["content"])
    return tweets


'''
    1.1 (3 points) Implement a simple tokenizer. All it does is extracting tokens based on white
    space (Use split() API in python; do not use any parameter when calling split).
'''
def simple_tokenizer(tweets):
    return [tweet.split() for tweet in tweets]


def analyzer(tokenizers, tweets):
    for token in tokenizers:
        print(token)
        # print(tokenizers[token]["tokens"][0])
        tokenizers[token]["types"] = [set(t) for t in tokenizers[token]["tokens"]]
        tokenizers[token]["num_types"] = [len(t) for t in tokenizers[token]["types"]]
        tokenizers[token]["num_tokens"] = [len(t) for t in tokenizers[token]["tokens"]]
        tokenizers[token]["ratio"] = [tokenizers[token]["num_types"][t]/tokenizers[token]["num_tokens"][t] for t in range(len(tweets))]
        print("Number of types: ", tokenizers[token]["num_types"])
        print("Number of tokens: ", tokenizers[token]["num_tokens"])
        print("type/token ratio: ", tokenizers[token]["ratio"])
    return tokenizers


'''
    1.2 (3 points) Use NLTK to tokenize a file. Your first step is to install NLTK library. Experiment
    with the WordPunctTokenizer in NLTK.3 Report the number of types and type/token ratio for each
    tokenizer
'''
def nltk_tokenizer(tweets):
    tokenizers = {"TreebankWordTokenizer": {"tokens": [TreebankWordTokenizer().tokenize(tweet) for tweet in tweets]},
                  "WordPunctTokenizer": {"tokens": [WordPunctTokenizer().tokenize(tweet) for tweet in tweets]},
                  "WhitespaceTokenizer": {"tokens": [WhitespaceTokenizer().tokenize(tweet) for tweet in tweets]},} 
    
    tokenizers = analyzer(tokenizers, tweets)
    return tokenizers


'''
    1.3 (6 points) Try another tokenization tookit: spaCy. Install spaCy in you environment4
    . spaCy by default goes through a bigger NLP pipeline than tokenization, you may want to disable tagger,
    parser, and ner to speed up the tokenization
    ( nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"]) ). 
    There are also interesting results such as lemmas in addition to tokens that you need for this assignment. You may
    want to check them out in your own interest.
'''
def spacy_tokenizer(tweets):
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"]) 
    toked = [[w.text for w in nlp(tweet)] for tweet in tweets]    
    stats = analyzer({"SpacyTokenizer": {"tokens": toked}}, tweets)
    return stats


'''
    1.4 (6 points) Build your own tokenzier with BPE model. Follow steps in https://github.com/
    huggingface/tokenizers. You will use the tweets in IRAhandle_tweets_1.csv as training data
    instead of using wiki.train(valid/test).raw (you may need to output a temporary file with a
    tweet per line for the training). After training, tokenize all the tweets and report the same three
    items
'''
def new_tokenizer(tweets):
    return 0

first_3_tweets = get_tweets(datafile)[:3]
# nltk_tokenizer(first_3_tweets)
# spacy_tokenizer(first_3_tweets)

'''
    1.5 (7 points) Compare the output of these tokenizers and discuss their pros and cons.
'''