import nltk
import langid
import re
import os
from timer import Timer

url_regex = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})'
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist
apple_tweets_path = './data/apple/tweets/'

unk_url = '<url>'

max_line = -1
num_tweet_ex = 5

def print_lines(lines):
    for line in lines:
        print(line)

def read_tweets_data():
    result = []
    for root, dirs, files in os.walk(apple_tweets_path):  
        for filename in files:
            with open(apple_tweets_path + filename, 'r', encoding='utf-8') as file:
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
        if not token.startswith('http') and \
            '.com' not in token:
            cur_tokenized.append(token)
        else:
            cur_tokenized.append(unk_url)
    if cur_tokenized[-1] == unk_url:
        cur_tokenized = cur_tokenized[:-1]
    return ' '.join(cur_tokenized)

# Clean text by removing unnecessary characters and altering the format of words.
def clean_text(text, whitelist):
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

    return text if len(text) > 0 else pre_text

def output_data(lines, date2stock):
    stock_keys = ['open', 'high', 'low', 'close', 'volume']
    with open('./data/apple/output.csv', 'w', encoding='utf-8') as file:
        file.write('date,open,high,low,close,volume,tweet\n')
        for date, text in lines:
            if date not in date2stock:
                continue
            stock_data = date2stock[date]
            tokens = tuple([date])
            tokens += tuple([stock_data[key] for key in stock_keys])
            tokens += tuple([text])
            file.write('{},{},{},{},{},{},{}\n'.format(*tokens))

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
        lines = [(date, clean_text(text, EN_WHITELIST)) for date, text in lines]
        print_lines(lines[:num_tweet_ex])

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
    


    
if __name__ == '__main__':
    process_data()