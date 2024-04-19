import os
import tensorflow as tf
import tensorflow_addons as tfa
import pickle as pkl
import text_file_operations as tfo
import text_neural_networks as tnn


def parse_split():
    english_list, eng_labels, spanish_list, span_labels, translations = tfo.parse_translations('English-Spanish.txt')
    (en_train_data, en_train_label,
     en_val_data, en_test_data,
     en_val_label, en_test_label) = tnn.split_data(english_list, eng_labels, 0.7)
    (sp_train_data, sp_train_label,
     sp_val_data, sp_test_data,
     sp_val_label, sp_test_label) = tnn.split_data(spanish_list, span_labels, 0.7)
    return ((en_train_data, en_train_label, en_val_data, en_test_data, en_val_label, en_test_label),
            (sp_train_data, sp_train_label, sp_val_data, sp_test_data, sp_val_label, sp_test_label))


def tokenize(en_train_data, sp_train_data):
    tokenizer_en = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_en.fit_on_texts(en_train_data)
    tokenizer_sp = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_sp.fit_on_texts(sp_train_data)
    en_train_seq = tokenizer_en.texts_to_sequences(en_train_data)
    sp_train_seq = tokenizer_sp.texts_to_sequences(sp_train_data)
    return tokenizer_en, tokenizer_sp, en_train_seq, sp_train_seq


def build_transformer(input_vocab_size, target_vocab_size, num_layers, model_embedding_dim,
                      num_heads, feed_forward_dim, position_encoding_input, positional_encoding_target, dropout_rate):
    inputs = tf.keras.layers.Input(shape=(None,))
    targets = tf.keras.layers.Input(shape=(None,))

    enc_padding_mask = tf.keras.layers.Lambda(tfa.seq2seq.layers.PaddingMask)(inputs)
    dec_padding_mask = tf.keras.layers.Lambda(tfa.seq2seq.layers.PaddingMask)(inputs)

    # Define the transformer model
    transformer = get_model(
        token_num=input_vocab_size,
        embed_dim=model_embedding_dim,
        encoder_num=num_layers,
        decoder_num=num_layers,
        head_num=num_heads,
        hidden_dim=feed_forward_dim,
        dropout_rate=dropout_rate,
        use_same_embed=False
    )

    output = transformer([inputs, targets])

    output = transformer(
        [inputs, targets],
        training=True,
        enc_padding_mask=enc_padding_mask,
        dec_padding_mask=dec_padding_mask
    )

    return tf.keras.Model(inputs=[inputs, targets], outputs=output)


def translator(num_layers=4, model_embedding_dim=128, num_heads=8, feed_forward_dim=512,
               positional_encoding_input=1000, positional_encoding_target=1000, dropout_rate=0.1,
               optim_learning_rate=0.001, optim_beta_1=0.9, optim_beta_2=0.98, optim_epsilon=1e-9,
               batch_size=64, epochs=10, validation_split=0.3):
    print('Parsing/Loading the data and splitting into\nTraining, Testing, and Validation Data ...')
    ((en_train_data, en_train_label, en_val_data, en_test_data, en_val_label, en_test_label),
     (sp_train_data, sp_train_label, sp_val_data, sp_test_data, sp_val_label, sp_test_label)) = parse_split()

    print('Tokenizing Data ...')
    tokenizer_en, tokenizer_sp, en_train_seq, sp_train_seq = tokenize(en_train_data, sp_train_data)
    input_vocab_size = len(tokenizer_en.word_index) + 1  # +1 to reserve index 0 for padding
    target_vocab_size = len(tokenizer_sp.word_index) + 1  # +1 to reserve index 0 for padding
    tokenizer_en_val, tokenizer_sp_test, en_test_seq, sp_test_seq = tokenize(en_test_data, sp_test_data)
    tokenizer_en_val, tokenizer_sp_val, en_val_seq, sp_val_seq = tokenize(en_val_data, sp_val_data)

    print('Building Transformer Model ...')
    transformer_model = build_transformer(input_vocab_size, target_vocab_size, num_layers,
                                          model_embedding_dim, num_heads, feed_forward_dim,
                                          positional_encoding_input, positional_encoding_target, dropout_rate)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=optim_learning_rate,
        beta_1=optim_beta_1,
        beta_2=optim_beta_2,
        epsilon=optim_epsilon
    )
    print('Compiling Transformer Model ...')
    transformer_model.compile(
        optimizer=optimizer,
        loss=loss_function
    )
    print('Fitting Transformer Model on Training Data...')
    transformer_model.fit(
        [en_train_seq, sp_train_seq],
        sp_train_seq,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split
    )

    print('Running Model on Validation Data ...')
    validation_loss = transformer_model.evaluate([en_val_seq, sp_val_seq], sp_val_seq)
    print("Validation Loss:", validation_loss)
    print('Testing Model ...')
    testing_predictions = transformer_model.predict([en_test_seq, sp_test_seq])
    true_positive, false_positive, false_negative, true_negative = tnn.tf_pos_neg(testing_predictions, sp_test_seq)
    tnn.nn_eval(true_positive, false_positive, false_negative, true_negative)
    pkl.dump(transformer_model, open('transformer_model.pkl', 'wb'))
    print('transformer_model.pkl saved to {}'.format(os.getcwd()))
    return transformer_model


def load_transformer_model():
    transformer_model = None
    pkl_boolean = False
    for file_name in next(os.walk(os.getcwd() + '/'))[2]:
        if file_name == 'transformer_model.pkl':
            pkl_boolean = True
    if pkl_boolean:
        print('.pkl file found! :D')
        transformer_model = pkl.load(open('translation_vars.pkl', 'rb'))
    elif not pkl_boolean:
        print('.pkl file not found :(')
    return transformer_model
