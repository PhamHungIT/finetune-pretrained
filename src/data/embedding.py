
import os

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedding:

    def __init__(self, type) -> None:
        """
        Initial an embedding class for converting sentence
        to vector embedding

        Args:
            type: (str) Type of embedding.
                Optional: 'bow', 'tf_idf' or 'word2vec'
        """
        self.type = type

    def fit(self, corpus, config):
        """
        Create transform for converting word to vector
        Args:
            corpus: (list) List of sentences
            config: (dict) param for vectorizer
        """
        self.corpus = corpus
        if self.type == 'word2vec':
            self.corpus = [s.split() for s in self.corpus]
            self.vectorizer = Word2Vec(
                sentences=self.corpus,
                vector_size=config['vector_size'],
                min_count=config['min_count'],
                workers=os.cpu_count() - 1
            )
            self.embedding_dim = config['vector_size']

        elif self.type == 'bow':
            self.vectorizer = CountVectorizer(min_df=config['min_count'])
            self.vectorizer.fit(self.corpus)
            self.embedding_dim = len(self.vectorizer.vocabulary_)

        elif self.type == 'tf_idf':
            self.vectorizer = TfidfVectorizer(min_df=config['min_count'])
            self.vectorizer.fit(self.corpus)
            self.embedding_dim = len(self.vectorizer.vocabulary_)

    def __call__(self, sentence):
        """
        Convert a sentence to vector respectively

        Args:
            sentence: (str) Sentence is given
        """
        if self.type == 'word2vec':
            words = sentence.split()
            embedding_sentence = []
            for word in words:
                try:
                    embedding_word = self.vectorizer.wv[word]
                    embedding_sentence.append(embedding_word)
                except KeyError:
                    # Word not in vocabulary
                    pass
            if len(embedding_sentence) == 0:
                embedding_sentence = np.zeros(self.embedding_dim)
            else:
                embedding_sentence = np.array(embedding_sentence).mean(axis=0)

        else:
            embedding_sentence = self.vectorizer.transform([sentence])
            embedding_sentence = embedding_sentence.toarray()[0]

        return embedding_sentence
