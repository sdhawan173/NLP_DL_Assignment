import text_functions as tfx
import text_file_operations as tfo
import text_neural_networks as tnn

print('---Loading and Preprocessing---')
positive_data = tfo.load_data('comments1k_pos')
negative_data = tfo.load_data('comments1k_neg')
data_list = (positive_data, negative_data)
all_data = data_list[0] + data_list[1]
data_labels = tfx.create_labels((data_list[0], data_list[1]))
data_split_words = tfx.split_words(all_data, exclusion=True)
data_split_stem = tfx.stemming(data_split_words)
print('Question 1: Sentiment Analysis--------')
word2vec_m3 = tfx.word2vec_cbow(data_split_words,
                                vector_size=100,
                                window=5,
                                min_count=1)
print('Part 1')

# c1 = tnn.word2vec_nn(data_split_stem,
#                      data_labels,
#                      word2vec_m3,
#                      verbose_boolean=True)
# c2 = tnn.word2vec_nn(data_split_stem,
#                      data_labels,
#                      word2vec_model=None,
#                      verbose_boolean=True)
print('Part 2')
# tnn.classifier(data_split_stem, data_labels, word2vec_m3, 'rnn')
# tnn.classifier(data_split_stem, data_labels, word2vec_m3, 'lstm')
# tnn.classifier(data_split_stem, data_labels, word2vec_m3, 'gru')
# tnn.stacked_bilstm(data_split_stem, data_labels, word2vec_m3)
print('Question 2: Text Translation--------')
english_list, eng_labels, spanish_list, span_labels, translations = tfo.parse_translations('English-Spanish.txt')
english_split_data = tnn.split_data(english_list, eng_labels, 0.7)
spanish_split_data = tnn.split_data(spanish_list, span_labels, 0.7)
sentence = ('Deep Learning is widely used in Natural Language '
            'Processing, as Dr. Sun said in CSC 495/693')

for eng, span in zip(english_list[0:300], spanish_list[200:500]):
    print('ENGLISH INPUT: {}, SPANISH INPUT: {}'.format(eng, span))
