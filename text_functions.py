import os
import time
import string
import subprocess
import shlex
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import text_file_operations as tfo


try:
    nltk.data.find('tokenizers/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.add('n\'t')
STOP_WORDS.add('\'s')
STOP_WORDS.add('\'m')
STOP_WORDS.add('\'ll')
STOP_WORDS.add('\'ve')
STOP_WORDS.add('like')
STOP_WORDS.add('good')
STOP_WORDS.add('also')
STOP_WORDS.add('every')

PUNCTUATION = set(string.punctuation)
PUNCTUATION.add('â–')
PUNCTUATION.add('br')
PUNCTUATION.add('<br>')
PUNCTUATION.add('<\\br>')
PUNCTUATION.add('``')
PUNCTUATION.add('/')
PUNCTUATION.add('\'\'')
PUNCTUATION.add('\"')
PUNCTUATION.add('..')
PUNCTUATION.add('...')
PUNCTUATION.add('....')
PUNCTUATION.add('.....')
PUNCTUATION.add('......')
PUNCTUATION.add('.......')
PUNCTUATION.add('........')
PUNCTUATION.add('.........')
PUNCTUATION.add('..........')
PUNCTUATION.add('...........')
PUNCTUATION.add('............')
PUNCTUATION.add('.............')
PUNCTUATION.add('..............')

PWD = os.getcwd()


def split_sentence(text_data):
    """
    Splits each comment from list of comments into sentences
    :param text_data: list of comments
    :return: list of lists of comments split into sentences
    """
    # Make sure 'punkt' is installed
    print('\nSplitting {} comments by sentences ...'.format(len(text_data)))
    text_split_sentence = text_data
    for index in range(len(text_data)):
        text_split_sentence[index] = nltk.sent_tokenize(text_data[index][0])
    return text_split_sentence


def replace_multiwords(word_list):
    """
    replaces individual words in list with words that may be multiwords
    :param word_list: list of words
    :return: list of words with multiword combinations replaced by a single string
    """
    finder = BigramCollocationFinder.from_words(word_list)
    finder.apply_freq_filter(3)
    collocations = finder.nbest(BigramAssocMeasures.pmi, 5)
    for collocation_to_merge in collocations:
        merged_words = []
        i = 0
        while i < len(word_list):
            if i < len(word_list) - 1 and (word_list[i], word_list[i + 1]) == collocation_to_merge:
                merged_words.append(' '.join(collocation_to_merge))
                i += 2
            else:
                merged_words.append(word_list[i])
                i += 1
        word_list = merged_words.copy()
    return word_list


def exclusion_filter(word_list):
    exclusion_list = []
    for word in word_list:
        if word.lower() not in STOP_WORDS and word.lower() not in PUNCTUATION:
            exclusion_list.append(word.lower())

    punc_list = list(PUNCTUATION)
    for word in exclusion_list:
        new_items = []
        word_index = None
        remove_boolean = None
        for character in word:
            if character in punc_list:
                word_index = exclusion_list.index(word)
                remove_boolean = True
                punc_index = punc_list.index(character)
                new_items = exclusion_list[word_index].split(punc_list[punc_index])

        if remove_boolean:
            exclusion_list.remove(exclusion_list[word_index])
            insert_count = 0
            for item in new_items:
                if item != '':
                    exclusion_list.insert(word_index + insert_count, item)
                    insert_count += 1
    return exclusion_list


def split_words(text_data, exclusion=True):
    """
    Splits each comment from list of comments into words
    :param text_data: list of comments
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :return: list of lists of comments split into words
    """
    print('Splitting {} comments by words ...'.format(len(text_data)))
    if exclusion:
        print('Excluding stop words and punctuation ...')
    text_split_word = []
    for comment in text_data:
        comment_words = []
        temp = []
        for sentence in comment:
            if not exclusion:
                temp = nltk.word_tokenize(sentence)
            elif exclusion:
                temp = exclusion_filter(nltk.word_tokenize(sentence))
            for word in temp:
                comment_words.append(word.lower())
        text_split_word.append(comment_words)
    return text_split_word


def lemmatization(text_split_sentence, exclusion=False, multiword=False):
    """
    lemmatizes each comment from list of comments into lemmatized words
    :param text_split_sentence: list of lists of comments split into sentences
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :param multiword: Boolean to activate replace_multiwords function
    :return: list of lists of comments split into lemmatized words
    """
    print('\nLemmatizing {} comments ... '.format(format(len(text_split_sentence))))
    if exclusion:
        print('Excluding stop words and punctuation ...')
    if multiword:
        print('Replacing multiwords ...')
    lemmatizer = WordNetLemmatizer()
    all_comments_lemmatized = []
    if not multiword:
        for comment in text_split_sentence:
            comment_lemmatized = []
            for sentence in comment:
                words = [word.lower() for word in word_tokenize(sentence)]
                if exclusion:
                    words = exclusion_filter(words.copy())
                for word in words:
                    comment_lemmatized.append(lemmatizer.lemmatize(word))
            all_comments_lemmatized.append(comment_lemmatized)
    elif multiword:
        for comment in text_split_sentence:
            word_list = []
            comment_lemmatized = []
            for sentence in comment:
                word_list += [word.lower() for word in word_tokenize(sentence)]
            word_list = exclusion_filter(word_list.copy())
            word_list = replace_multiwords(word_list.copy())
            for word in word_list:
                comment_lemmatized.append(lemmatizer.lemmatize(word))
            all_comments_lemmatized.append(comment_lemmatized)
    return all_comments_lemmatized


def stemming(text_split_word, exclusion=False, multiword=False):
    """
    stems each comment from list of comments into stemmed words
    :param text_split_word: list of lists of comments split into sentences
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :param multiword: Boolean to activate replace_multiwords function
    :return: list of comments split into stemmed words
    """
    print('Stemming {} comments ... '.format(format(len(text_split_word))))
    if exclusion:
        print('Excluding stop words and punctuation ...')
    if multiword:
        print('Replacing multiwords ...')
    stemmer = nltk.PorterStemmer()
    all_comments_stemmed = []
    for word_list in text_split_word:
        stemmed_comment = [stemmer.stem(word) for word in word_list]
        if exclusion:
            stemmed_comment = exclusion_filter(stemmed_comment.copy())
        if multiword:
            stemmed_comment = replace_multiwords(stemmed_comment.copy())
        all_comments_stemmed.append(stemmed_comment)
    return all_comments_stemmed


def lda(all_comments, n):
    """

    :param all_comments:
    :param n:
    :return:
    """
    print('\nPerforming Latent Drichlet Allocation ({} topics)...'.format(n))
    lda_dict = corpora.Dictionary(all_comments)
    lda_corpus = []
    for comment in all_comments:
        lda_corpus.append(lda_dict.doc2bow(comment))
    lda_model = LdaModel(lda_corpus, n, id2word=lda_dict)
    for output in lda_model.print_topics():
        print(output)
    return lda_model


def word2vec_cbow(data, vector_size, window, min_count, sg=0, epochs=30, verbose=True):
    if verbose:
        print('Running word2vec with CBOW\n'
              '(vector_size={}, window={}, min_count={}) ...'.format(vector_size, window, min_count))
    start = time.time()
    # sg=0 is CBOW
    word2vec_model = Word2Vec(data, vector_size=vector_size, window=window, min_count=min_count, sg=sg, epochs=epochs)
    end = time.time()
    total = '{:.3f}'.format(end - start)
    if verbose:
        print('Time to train: {} seconds'.format(total))
    return word2vec_model


def load_glove_model():
    return KeyedVectors.load_word2vec_format(PWD + '/glove/vectors.txt', binary=False, no_header=True)


def model_eval(eval_list, similarity_size, word2vec_model=None, glove_model=None):
    if word2vec_model is not None:
        print('Evaluating word2vec model ...')
    if glove_model is not None:
        print('Evaluating glove model ...')
    for word in eval_list:
        similar_words = ''
        if word2vec_model is not None:
            similar_words = word2vec_model.wv.most_similar(word.lower(), topn=similarity_size)
        if glove_model is not None:
            similar_words = glove_model.most_similar(word.lower(), topn=similarity_size)
        similar_words_output = [
            (output_word, '{:.2f}'.format(similarity * 100))
            for output_word, similarity in similar_words
        ]
        print('Words similar to \'{}\':{}'.format(word, similar_words_output))


def no_space_path(path):
    if path.__contains__(' '):
        path_components = path.split('/')
        path_components_escaped = [
            shlex.quote(component)
            if ' ' in component
            else component
            for component in path_components
        ]
        path = '/'.join(path_components_escaped)
        return path


def glove_world(data, shell_file='demo.sh', text_file_name='all_comments'):
    print('Glove main function ...')
    glove_pwd = os.getcwd() + '/' + 'glove'
    os.chdir(glove_pwd)

    print('Saving data for glove to use ...')
    tfo.save_data(data, PWD + '/glove/' + text_file_name + '.txt')

    print('Running \'./{}\'\n'
          '(corpus=all_comments.txt, vector_size=100, window=5, min_count=1) ...'.format(shell_file))
    # chmod +x demo.sh
    # chmod +x ... path ... /glove/*  spaces in path must be preceded with escape character \
    shell_pwd = os.path.join(PWD, 'glove/')
    print(shell_pwd)
    os.chdir(shell_pwd)
    shell_pwd = no_space_path(shell_pwd)
    shell_path = shell_pwd + shell_file
    start = time.time()
    subprocess.run([shell_path], shell=True)
    end = time.time()
    total = '{:.3f}'.format(end - start)
    print('Time to train: {} seconds'.format(total))
    os.chdir(PWD)

    print('Loading glove model ...')
    glove_model = load_glove_model()
    return glove_model


def vector_visualize(model, word2vec=False, glove=False, top_count=25, show=False, save=True):
    embeddings = None
    words = None
    if word2vec:
        embeddings = [model.wv[word] for word in model.wv.index_to_key][:top_count]
        words = list(model.wv.index_to_key)[:top_count]
    if glove:
        embeddings = [model[word] for word in model.index_to_key][:top_count]
        words = list(model.index_to_key)[:top_count]

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=15)

    model_name = ''
    if word2vec:
        model_name = 'word2vec'
    if glove:
        model_name = 'glove'
    title = '{} PCA Visualization of Top {} Word Embeddings'.format(model_name, top_count)

    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    if show:
        plt.show()
    if save:
        plt.savefig(PWD + '/CSC 693 Assignment 2 Writeup/{} Top {} Vectors.png'.format(model_name, top_count))
    plt.clf()
    plt.close()


def create_labels(data_arrays):
    print('Creating data labels ...')
    labels = []
    for i in range(len(data_arrays)):
        labels += [i for _ in data_arrays[i]]
    return labels


def comment_embeddings(data, data_labels, model):
    comment_embedding_data = []
    new_labels = []
    word_embedding_data = {word: model.wv[word] for word in model.wv.index_to_key}
    for comment, label in zip(data, data_labels):
        embedding_sum = [0] * len(model.wv[model.wv.index_to_key[0]])
        word_count = 0
        for word in comment:
            if isinstance(word, str) and word in word_embedding_data:
                embedding_sum = [x + y for x, y in zip(embedding_sum, word_embedding_data[word])]
                word_count += 1
        if word_count > 0:
            embedding_avg = [x / word_count for x in embedding_sum]
            comment_embedding_data.append(embedding_avg)
            new_labels.append(label)
    return np.array(comment_embedding_data), np.array(new_labels)
