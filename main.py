import codecs
import re
from nltk.stem.porter import PorterStemmer
import gensim


def preprocessing(paragraphs):
    paragraphs = paragraphs.split("\r\n\r\n")
    original_paragraphs = []
    i = 0
    while i < len(paragraphs):
        if "gutenberg" in paragraphs[i].lower():  # Remove Gutenberg-paragraphs
            paragraphs.pop(i)
            continue
        # No more paragraphs will be removed, so it is safe to store the original ones in an array
        # and access them by index later
        original_paragraphs.append(paragraphs[i])
        paragraphs[i] = re.sub(r'[^\w\s]|[\n\t\r]', "", paragraphs[i].lower())  # Remove punctuation and whitespace
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
    return bag_of_words, dictionary


def retrieval_models(bow_corpus, dictionary):
    # Train the tfidf model on our bow corpus
    tfidf_model = gensim.models.TfidfModel(bow_corpus)
    # Transform the whole corpus via TF-IDF
    tfidf_corpus = tfidf_model[bow_corpus]
    # Index the TF-IDF corpus in preparation for querying
    tfidf_index = gensim.similarities.MatrixSimilarity(tfidf_corpus, num_features=len(dictionary))
    # Doing the same procedure for LSI
    lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
    lsi_corpus = lsi_model[bow_corpus]
    lsi_index = gensim.similarities.MatrixSimilarity(lsi_corpus, num_features=len(dictionary))
    print("First 3 topics: ")
    topics = lsi_model.show_topics(3)
    for topic in topics:
        print(topic)
    return tfidf_index, lsi_index, tfidf_model, lsi_model


def query(original_paragraphs, query_sentence, dictionary, tfidf_index, lsi_index, tfidf_model, lsi_model):
    # Preprocessing the query, and transforming it into a list of stemmed words
    _, query_bow = preprocessing(query_sentence)
    # Converting the list of words into a bag of words representation
    query_bow = dictionary.doc2bow(query_bow[0])
    # Convert query bow to TF-IDF representation
    tfidf_query_representation = tfidf_model[query_bow]
    # Find similarities using the previously generated TF-IDF index for our document collection
    similarities = enumerate(tfidf_index[tfidf_query_representation])
    # Sort similarities, most relevant documents will appear first
    similarities = list(sorted(similarities, key=lambda x: x[1], reverse=True))
    DOCS_TO_RETRIEVE = 3
    for i in range(DOCS_TO_RETRIEVE):
        # Shorten paragraph to 5 lines
        shortened_original = "\n".join(original_paragraphs[similarities[i][0]].split('\n')[0:5])
        print(f"Document {similarities[i][0]}, {round(similarities[i][1] * 100, 2)}"
              f"% similarity:\n{shortened_original}\n")
    # Convert query to LSI representation
    lsi_query = lsi_model[tfidf_query_representation]
    # Find the top 3 LSI topics:
    most_relevant_topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:DOCS_TO_RETRIEVE]
    print("3 most relevant topics: ")
    for topic_weight_pair in most_relevant_topics:
        topic_idx = topic_weight_pair[0]
        topic = lsi_model.show_topic(topic_idx)
        print(f"Topic {topic_idx}: {topic}")
    # Find query-document-similarities and retrieve 3 most relevant documents
    similarities = enumerate(lsi_index[lsi_query])
    similarities = sorted(similarities, key=lambda kv: -kv[1])[:DOCS_TO_RETRIEVE]
    print("\nTop 3 documents:")
    for similarity in similarities:
        shortened_original = "\n".join(original_paragraphs[similarity[0]].split('\n')[0:5])
        print(f"Document {similarity[0]}, {round(similarity[1] * 100, 2)}"
              f"% similarity:\n{shortened_original}\n")


def main():
    QUERY = "What is the function of money?"  # Change this if you want to test the program

    f = codecs.open("pg3300.txt", "r", "utf-8")
    paragraphs = f.read()
    f.close()
    original_paragraphs, paragraphs = preprocessing(paragraphs)
    # print(paragraphs[0:5])
    corpus, dictionary = dictionary_building(paragraphs)
    tfidf_index, lsi_index, tfidf_model, lsi_model = retrieval_models(corpus, dictionary)
    query(original_paragraphs, QUERY, dictionary, tfidf_index, lsi_index, tfidf_model,
          lsi_model)


if __name__ == '__main__':
    main()
