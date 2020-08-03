import nltk
import sys
import string
import math
import os 
FILE_MATCHES = 4
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


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data={}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename),encoding='utf8') as f:
            data[filename]=f.read()
    return data        
        


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    contents = []
    contents.extend(
        word.lower() for word in
        nltk.word_tokenize(document)
        if any((c not in string.punctuation and word not in nltk.corpus.stopwords.words("english")) for c in word)  
    )
    return contents


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    new_list=[]
    length=len(documents)
    for key in documents:
        for item in documents[key]:
            new_list.append(item)
    count=0
    idf_dict={}
    for word in new_list:
        count=0
        for doc in documents:
            if word in documents[doc]:
                count+=1
        idf=math.log(length/count)
        idf_dict[word]=idf
    
    return idf_dict

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    query=list(query)
    file_rank={}
    for file in files:
        tf=0
        idf=0
        tf_idf=0
        for word in query:
            if word in files[file]:
                tf=(files[file]).count(word)
                idf=idfs[word]
                tf_idf=tf_idf+tf*idf
        file_rank[file]=tf_idf
    file_list=sorted(file_rank.keys(),key=file_rank.get,reverse=True)
    return file_list[:n]
    
    
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    query=list(query)
    sent_rank={}
    for sent in sentences:
        idf=0
        mwm=0
        count=0
        value=[]
        for word in query:
            if word in sentences[sent]:
                idf=idfs[word]
                count+=sentences[sent].count(word)
                mwm=mwm+idf
        value.append(mwm)
        value.append(count/len(sentences[sent]))        
        sent_rank[sent]=value
    #print(sent_rank)    
    sent_list=sorted(sent_rank, key=lambda k: (sent_rank[k][0], sent_rank[k][1]),reverse=True)
    
    return sent_list[:n]


if __name__ == "__main__":
    main()
