import nltk
import sys
import os
import string
from nltk.sem.logic import QuantifiedExpression
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    askquestion(file_idfs, file_words, files)

def askquestion(file_idfs, file_words, files):

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)

    askquestion(file_idfs, file_words, files)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    filenames = {}

    path = directory + os.sep

    for filename in os.listdir(path): 
        if os.path.isfile(os.path.join(path, filename)):
            with open(os.path.join(path, filename), encoding="utf8") as f:
                filenames[filename] = f.read()

    print("loading files...")
    return filenames


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    
    """
    stop_words = nltk.corpus.stopwords.words("english")

    tokenizer = nltk.word_tokenize(document.lower())

    return [word for word in tokenizer if word not in string.punctuation and word not in stop_words]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfvalues = {}

    numberofdoc = len(documents)

    counter = 0

    for words in documents.values(): 
        for word in words: 
            for words2 in documents.values():
                if word in words2: 
                    counter += 1 
            idfvalues[word] = math.log(numberofdoc/counter)
            counter = 0


    return idfvalues
        
def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidf = {}

    for file in files.keys(): 
        tfidf[file] = 0

    #for each in the query set
    for word in query:
        for file, values in files.items(): 
            if word in values: 
                tfidf[file] += idfs[word] * values.count(word)

    querymatch = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)

    return ([element[0] for element in querymatch][:n])


        


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    sentencesvalues = {}

    for sentence, words in sentences.items(): 
        querywords = query.intersection(words)

        value = 0 
        for word in querywords: 
            value += idfs[word]

        numwordsinquery = sum(map(lambda x: x in querywords, words))

        query_term_density = numwordsinquery/ len(words)

        sentencesvalues[sentence] = {
            'idf': value,
            'qtd': query_term_density, 
        }

    ranked_sentences = sorted(sentencesvalues.items(), key=lambda x: (x[1]['idf'], x[1]['qtd']), reverse=True)
    ranked_sentences = [x[0] for x in ranked_sentences]

    return ranked_sentences[:n]




if __name__ == "__main__":
    main()
