
import langid
import re
import os
from timer import Timer
from time import time
import json
from functools import reduce
import argparse
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist

unk_url = '<url>'

min_word_count = 3
max_line = -1
num_tweet_ex = 5

def add_arguments(parser):
    parser.add_argument("company",
                        help="The name of company")
    parser.add_argument("-s", "--no-stopwords",
                        help="Remove English stopwords",
                        action="store_true")

def print_lines(lines):
    for line in lines:
        print(line)

def read_tweets_data():
    result = []
    for root, dirs, files in os.walk(tweets_path):  
        for filename in files:
            with open(tweets_path + filename, 'r', encoding='utf-8') as file:
                # remove first line
                next(file)
                idx = 0
                for line in file:
                    line = line.strip()
                    if max_line != -1 and idx >= max_line:
                        break
                    tokens = line.split(';')
                    username, date, retweets, favorites = tokens[:4]
                    rest = tokens[4:]
                    text = ';'.join(rest[:len(rest) - 5])

                    lang = langid.classify(text)
                    if lang[0] != 'en':
                        continue
                    idx += 1
                    try:
                        if text[0] == '"':
                            text = text[1:]
                        if text[-1] == '"':
                            text = text[:-1]
                    except IndexError:
                        print(tokens)
                        raise IndexError
                    result.append((date[:10], text))
    return result

def read_stock_data():
    result = {}
    with open('data/apple/stock_data.csv', 'r', encoding='utf-8') as file:
        next(file)
        idx = 0
        for line in file:
            if idx > 5:
                # break
                pass
            idx += 1
            line = line.strip()
            date, open_price, high, low, close, volume = line.split(',')
            result[date] = {'open': float(open_price), 'high': float(high), 
                            'low': float(low), 'close': float(close), 
                            'volume': float(volume)}
    return result

def remove_url(text):
    cur_tokenized = []
    tokenized = text.split()
    for token in tokenized:
        if '.com' not in token and \
            'http' not in token:
            cur_tokenized.append(token)
        else:
            cur_tokenized.append(unk_url)
    if cur_tokenized[-1] == unk_url:
        cur_tokenized = cur_tokenized[:-1]
    return ' '.join(cur_tokenized)

# return whether the given string is float
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Clean text by removing unnecessary characters and altering the format of words.
def clean_text(text, whitelist, no_stopwords=False):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"'cause", "because", text)
    text = re.sub(r"&amp;", "and", text)
    pre_text = text
    # text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = ''.join([ch for ch in text if ch in whitelist]).strip()
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    # stemming
    stemmer = PorterStemmer()
    if no_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([stemmer.stem(lemmatizer.lemmatize(word)) for word in text.split() if word not in stop_words])
    else:
        text = ' '.join([stemmer.stem(lemmatizer.lemmatize(word)) for word in text.split()])

    return text if len(text) > 0 else pre_text

def count_word(lines):
    word_count = {}
    total_word_count = 0
    for _, text in lines:
        for word in text.split():
            total_word_count += 1
            if is_number(word):
                continue
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
    return word_count, total_word_count

def output_data(lines, date2stock):
    def write_tokens(file, date, all_text):
        stock_data = date2stock[date]
        tokens = [date]
        tokens += [stock_data[key] for key in stock_keys]
        tokens += [' '.join(all_text)]
        file.write('{},{},{},{},{},{},{}\n'.format(*tokens))
    stock_keys = ['open', 'high', 'low', 'close', 'volume']
    with open(OUT_DIR + 'output.csv', 'w', encoding='utf-8') as file:
        file.write('date,open,high,low,close,volume,tweet\n')
        prev_date = lines[0][0]
        all_text = []
        for date, text in lines:
            if date not in date2stock:
                continue
            if date != prev_date:
                write_tokens(file, prev_date, all_text)
                prev_date = date
                all_text = [text]
            else:
                all_text.append(text)
        write_tokens(file, prev_date, all_text)
    with open(OUT_DIR + 'output_single_tweet.csv', 'w', encoding='utf-8') as file:
        file.write('date,open,high,low,close,volume,tweet\n')
        for date, text in lines:
            if date not in date2stock:
                continue
            stock_data = date2stock[date]
            tokens = [date]
            tokens += [stock_data[key] for key in stock_keys]
            tokens += [text]
            file.write('{},{},{},{},{},{},{}\n'.format(*tokens))

def output_vocab(vocab_set):
    idx = 0
    word2idx = {}
    for word in vocab_set:
        word2idx[word] = idx
        idx += 1
    with open(OUT_DIR + 'vocab.json', 'w', encoding='utf-8') as file:
        json.dump(word2idx, file)

def process_data():
    timer = Timer()
    with timer:
        print('Loading raw tweet data...')
        lines = read_tweets_data()
        print('Number of tweet data loaded: {}'.format(len(lines)))
        print_lines(lines[:num_tweet_ex])

    with timer:
        print('\nRemoving url...')
        lines = [(date, remove_url(text)) for date, text in lines]
        print_lines(lines[:num_tweet_ex])

    with timer:
        print('\nClean text...')
        # only keep alphabet and numbers
        lines = [(date, clean_text(text, EN_WHITELIST, no_stopwords=args.no_stopwords)) for date, text in lines]
        print_lines(lines[:num_tweet_ex])
        # sort by date
        lines = sorted(lines, key=lambda x : x[0])
        # vocab
        word_count, total_word_count = count_word(lines)
        vocab_set = {word for word in word_count \
                        if word_count[word] > min_word_count}
        used_vocab_count = reduce(lambda x, item: x + item[1] \
                        if item[0] in vocab_set else x, word_count.items(), 0)
        print('Total number of words: {}'.format(len(word_count)))
        print('Vocab size: {}'.format(len(vocab_set)))
        print('{:.2%} of words in vocab.'.format(used_vocab_count / total_word_count))


    with timer:
        print('\nLoading stock data...')
        date2stock = read_stock_data()
        print('Number of stock data loaded: {}'.format(len(date2stock)))
        print('Example of stock data:')
        print({'1980-12-19': date2stock['1980-12-19'],
               '2016-01-21': date2stock['2016-01-21']})

    with timer:
        print('\nOutputting data...')
        output_data(lines, date2stock)
        output_vocab(vocab_set)

    print('\nTotal time take: {}s'.format(time() - timer.start_time))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    OUT_DIR = './data/{}/'.format(args.company)
    tweets_path = './data/{}/tweets/'.format(args.company)
    process_data()