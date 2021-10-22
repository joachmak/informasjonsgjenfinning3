import codecs
import re
from nltk.stem.porter import PorterStemmer
import gensim


def preprocessing(paragraphs):
    paragraphs = paragraphs.lower().split("\r\n\r\n")
    i = 0
    while i < len(paragraphs):
        if "gutenberg" in paragraphs[i]:  # Remove Gutenberg-paragraphs
            paragraphs.pop(i)
            continue
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
    return paragraphs


def dictionary_building(paragraph_list):
    dictionary = gensim.corpora.Dictionary(paragraph_list)
    print(dictionary.keys())
    i = 1
    print(f"The term \"{dictionary.get(i)}\" occurs in the text {dictionary.dfs.get(i)} times")


def main():
    f = codecs.open("pg3300.txt", "r", "utf-8")
    paragraphs = f.read()
    f.close()
    paragraphs = preprocessing(paragraphs)
    dictionary_building(paragraphs)



if __name__ == '__main__':
    main()
