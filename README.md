# nlp_final
Fall 2024 nlp final - Joshua Lee and Yinan Gao
GitHub repo: https://github.com/zlmao666/nlp_final

# introduction
Our project centers on Named Entity Recognition. We implement a naive algorithm which parses through one or more documents and returns the NER category associated with each token. The training and test sets used are from the [ConLL2003 NER dataset](https://www.clips.uantwerpen.be/conll2003/ner/). Our algorithm is trained on the training set we compare it to spacy's NER algorithm using the test set. The dataset is taken from the [Reuters Corpus](https://trec.nist.gov/data/reuters/reuters.html), which contains 810,000 Reuters articles.

# instructions
Required libraries: spacy, numpy, collections, gensim.
All necessary data and files are included in the .zip file or can be obtained by cloning the GitHub repo.
To run the comparison between our naive NER algorithm and spacy's, simply run Evaluation<span>.</span>py.
To run the naive model on a new set of text, input your text into input.txt and run NER_naive.py.