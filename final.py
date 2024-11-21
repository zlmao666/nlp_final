import spacy
import gensim.downloader
import numpy as np
import time





def main():
    print('Loading model...')
    start = time.time()
    # model = gensim.downloader.load('word2vec-google-news-300') # This one is slower to load
    # model2 = gensim.downloader.load('glove-twitter-50')
    model = gensim.downloader.load('glove-wiki-gigaword-50') # This one is faster to load
    print('Done. ({} seconds)'.format(time.time() - start))

if __name__ == "__main__":
    main()