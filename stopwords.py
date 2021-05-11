
def load_stopwords():
    with open('stopwords') as f:
        stopwords = f.read()

    stopwords = stopwords.splitlines()
    return stopwords


def remove_stopwords(corpus, stopwords):
    for x in corpus:
        for word in x:
            if x in stopwords:
                x.remove(word)
