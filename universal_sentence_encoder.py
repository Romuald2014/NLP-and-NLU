import numpy as np
import pandas as pd
import nltk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

nltk.download('punkt')
# import tensorflow as tf
import tensorflow_hub as hub
from scipy import spatial

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


def use_segmenter(long_string, shuffle=True, min_segments=2, max_segments=20):
    """
  Return list of segments from input long_string
  The shuffle parameter specifies that sentences can change their order
  """

    if len(long_string) < 160:
        return {'Segment1': long_string}
    else:
        # Initial split into paragraphs
        corpus = long_string.split('\n\n')
        if len(corpus) < 2:
            # split into sentences
            corpus = nltk.sent_tokenize(long_string)

        # Corpus embedding
        corpus_use = embed(corpus)

        # Finding best number of segments
        sil_score_max = -1  # max siluet score for fine tuning (range -1, 1)
        if max_segments > len(corpus_use) - 1:  # Set limit on max labels
            max_segments = len(corpus_use) - 1

        best_n_clusters = min_segments  # initial
        for n_clusters in range(min_segments, max_segments):
            model = KMeans(n_clusters=n_clusters, max_iter=100, n_init=1)
            labels = model.fit_predict(corpus_use)
            sil_score = silhouette_score(corpus_use, labels)
            if sil_score > sil_score_max:
                sil_score_max = sil_score
                best_n_clusters = n_clusters

        # Clustering
        Kmean = KMeans(n_clusters=best_n_clusters)
        Kmean.fit(corpus_use)

        # Creating output list of segments
        segmented_corpus = {}

        if shuffle:
            # Create output segments list with phrases shuffled
            for cluster in range(Kmean.n_clusters):
                cluster_string = ''
                for i, label in enumerate(Kmean.labels_):
                    if cluster == label:
                        cluster_string += (corpus[i] + ' ')
                segmented_corpus['Segment' + str(cluster + 1)] = cluster_string
        else:
            # Create output segments list with phrases NOT shuffled
            current_label = Kmean.labels_[0]
            cluster_string = ''
            for i, label in enumerate(Kmean.labels_):
                if label == current_label:
                    cluster_string += (corpus[i] + ' ')
                else:
                    segmented_corpus['Segment' + str(i + 1)] = cluster_string
                    cluster_string = (corpus[i] + ' ')
                    current_label = label
            segmented_corpus['Segment' + str(i + 1)] = cluster_string

        return segmented_corpus
