import text_functions as tfx
import text_file_operations as tfo

print('---Loading and Preprocessing---')
positive_data = tfo.load_data('comments1k_pos')
negative_data = tfo.load_data('comments1k_neg')
data_list = (positive_data, negative_data)
all_data = data_list[0] + data_list[1]
data_labels = tfx.create_labels((data_list[0], data_list[1]))
data_split_words = tfx.split_words(all_data, exclusion=True)
data_split_stem = tfx.stemming(data_split_words)


