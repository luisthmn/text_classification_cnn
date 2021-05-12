
def load_stopwords():
    with open('stopwords') as f:
        stopwords = f.read()

    stopwords = stopwords.splitlines()
    return stopwords


def remove_stopwords(corpus, stopwords):
    new_corpus = []
    for x in corpus:
        words = [word for word in x.split() if not word in stopwords]
        x = ""
        for i in words:
            x += i + " "
        new_corpus.append(x)
    return new_corpus
