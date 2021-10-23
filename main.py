import codecs
import re
from nltk.stem.porter import PorterStemmer
import gensim


def preprocessing(paragraphs):
    paragraphs = paragraphs.lower().split("\r\n\r\n")
    original_paragraphs = []
    i = 0
    while i < len(paragraphs):
        if "gutenberg" in paragraphs[i]:  # Remove Gutenberg-paragraphs
            paragraphs.pop(i)
            continue
        # No more paragraphs will be removed, so it is safe to store the original ones in an array
        # and access them by index later
        original_paragraphs.append(paragraphs[i])
        paragraphs[i] = re.sub(r'[^\w\s]|[\n\t\r]', "", paragraphs[i])  # Remove punctuation and whitespace
        words = paragraphs[i].split(" ")
        words = list(filter(lambda word: word != "", words))
        paragraphs[i] = words
        i += 1
    # stem words
    stemmer = PorterStemmer()
    for paragraph in paragraphs:
        for i in range(len(paragraph)):
            paragraph[i] = stemmer.stem(paragraph[i])
    return original_paragraphs, paragraphs


def dictionary_building(paragraph_list):
    # dictionary.dfs.get(i) tells us how many times word at index i occurs in the text
    dictionary = gensim.corpora.Dictionary(paragraph_list)
    # Filter out stopwords
    stopwordfile = open("stopwords.txt", "r")
    stopwords = stopwordfile.read().split(",")
    stopwordfile.close()
    for i in range(len(stopwords)):
        if stopwords[i] in dictionary.token2id:
            stopwords[i] = dictionary.token2id[stopwords[i]]
    dictionary.filter_tokens(bad_ids=stopwords)
    # bag_of_words = [[(word_token_id, word_count), ...], ...]
    bag_of_words = []
    for paragraph in paragraph_list:
        bag_of_words.append(dictionary.doc2bow(paragraph))
    return bag_of_words


def main():
    f = codecs.open("pg3300.txt", "r", "utf-8")
    paragraphs = f.read()
    f.close()
    original_paragraphs, paragraphs = preprocessing(paragraphs)

    dictionary_building(paragraphs)


if __name__ == '__main__':
    main()
