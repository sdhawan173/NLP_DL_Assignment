import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sklms
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import text_functions as tfx

PWD = os.getcwd()


def tf_pos_neg(testing_predictions, testing_labels):
    prediction_labels = [1 if prediction > 0.5 else 0 for prediction in testing_predictions]
    true_positives_list = [(p == 1 and t == 1) for p, t in zip(prediction_labels, testing_labels)]
    false_positive_list = [(p == 1 and t == 0) for p, t in zip(prediction_labels, testing_labels)]
    false_negative_list = [(p == 0 and t == 1) for p, t in zip(prediction_labels, testing_labels)]
    true_negative_list = [(p == 0 and t == 0) for p, t in zip(prediction_labels, testing_labels)]
    true_positive_sum = sum(true_positives_list)
    false_positive_sum = sum(false_positive_list)
    false_negative_sum = sum(false_negative_list)
    true_negative_sum = sum(true_negative_list)
    return true_positive_sum, false_positive_sum, false_negative_sum, true_negative_sum


def nn_eval(true_positive, false_positive, false_negative, true_negative):
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('Accuracy  = ', accuracy)
    print('Precision = ', precision)
    print('Recall    = ', recall)
    print('F1        = ', f1)
    return accuracy, precision, recall, f1


def word2vec_nn(data, data_labels, word2vec_model, verbose_boolean=False):
    if verbose_boolean:
        print('Running word2vec Neural Network ...')
        print('     Aggregating word embeddings ...')
    if word2vec_model is not None:
        data, data_labels = tfx.comment_embeddings(data, data_labels, word2vec_model)
    elif word2vec_model is None:
        tokenizer = Tokenizer()
        # Fit the tokenizer on the list of comments
        tokenizer.fit_on_texts(data)

        # Convert comments into sequences of integers
        sequences = tokenizer.texts_to_sequences(data)

        # Pad sequences to ensure uniform length
        data = pad_sequences(sequences)
        data_labels = np.array(data_labels)
    print(data.shape)
    print(data_labels.shape)
    if verbose_boolean:
        print('     Creating training, validation, and testing data and labels ...')
    training_data, testing_data, training_labels, testing_labels = (
        sklms.train_test_split(
            data,
            data_labels,
            test_size=0.2,
            stratify=data_labels,
            random_state=666
        )
    )
    validation_data, testing_data = sklms.train_test_split(
        testing_data,
        test_size=0.5,
        random_state=666
    )
    validation_labels, testing_labels = sklms.train_test_split(
        testing_labels,
        test_size=0.5,
        random_state=666
    )
    if verbose_boolean:
        print('     Building neural network ...')
    neural_network = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=10,
            activation='relu'
        ),
        tf.keras.layers.Dense(
            units=1,
            activation='sigmoid'
        )
    ])
    print('     Compiling neural network ...')
    neural_network.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    if verbose_boolean:
        print('     Running neural network ...')
    neural_network.fit(
        training_data,
        training_labels,
        validation_data=(
            validation_data,
            validation_labels
        ),
        epochs=10,
        batch_size=32,
        verbose=0
    )
    if verbose_boolean:
        print('     Evaluating neural network ...')
    testing_predictions = neural_network.predict(testing_data)
    tp, fp, fn, tn = tf_pos_neg(testing_predictions, testing_labels)
    accuracy, precision, recall, f1 = nn_eval(tp, fp, fn, tn)
    return accuracy, precision, recall, f1


def w2v_nn_testing(data, data_labels, max_vector_size):
    print('Running Neural Network for vector sizes from 1 to {}'.format(max_vector_size))
    size_list = [i + 1 for i in range(max_vector_size)]
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for vector_size in size_list:
        print('Current vector size =', vector_size)
        word2vec_testing = tfx.word2vec_cbow(data, vector_size=vector_size, window=5, min_count=1, verbose=False)
        accuracy, precision, recall, f1 = word2vec_nn(data, data_labels, word2vec_testing, verbose_boolean=False)
        accuracy_list.append(100 * accuracy)
        precision_list.append(100 * precision)
        recall_list.append(100 * recall)
        f1_list.append(100 * f1)
    fig, ax = plt.subplots()
    ax.plot(size_list, accuracy_list, marker='o', label='Accuracy', zorder=4)
    ax.plot(size_list, precision_list, marker='o', label='Precision', zorder=3)
    ax.plot(size_list, recall_list, marker='o', label='Recall', zorder=2)
    ax.plot(size_list, f1_list, marker='o', label='F1 Score', zorder=1)
    plt.xlabel('Vector Size')
    plt.ylabel('Percentage')
    plt.title('Word2Vec Metrics vs. Vector Size')
    plt.legend()
    plt.show()
    fig.savefig(PWD + '/CSC 693 Assignment 2 Writeup/w2v_nn_vector_size_plot.png')
    plt.clf()
    plt.close()


def split_data(data, data_labels, train_test_size, test_val_size=0.5):
    training_data, testing_data, training_labels, testing_labels = (
        sklms.train_test_split(
            data,
            data_labels,
            test_size=train_test_size,
            stratify=data_labels,
            random_state=666
        )
    )
    validation_data, testing_data, validation_labels, testing_labels = (
        sklms.train_test_split(
            testing_data,
            testing_labels,
            test_size=test_val_size,
            random_state=666
        )
    )
    return training_data, testing_data, training_labels, testing_labels, validation_data, testing_data, validation_labels, testing_labels


def classifier(data, data_labels, word2vec_model, type_name, epochs=50, batch_size=64, learning_rate=0.001):
    print('Running Classifier Code for {}...'.format(type_name.upper()))
    print('... Preparing Data')
    data, data_labels = tfx.comment_embeddings(data, data_labels, word2vec_model)

    training_data, testing_data, training_labels, testing_labels = (
        sklms.train_test_split(
            data,
            data_labels,
            test_size=0.2,
            stratify=data_labels,
            random_state=666
        )
    )
    validation_data, testing_data, validation_labels, testing_labels = (
        sklms.train_test_split(
            testing_data,
            testing_labels,
            test_size=0.5,
            random_state=666
        )
    )

    training_data_tensor = torch.tensor(training_data, dtype=torch.float32)
    training_labels_tensor = torch.tensor(training_labels, dtype=torch.float32)
    validation_data_tensor = torch.tensor(validation_data, dtype=torch.float32)
    validation_labels_tensor = torch.tensor(validation_labels, dtype=torch.float32)
    testing_data_tensor = torch.tensor(testing_data, dtype=torch.float32)
    testing_labels_tensor = torch.tensor(testing_labels, dtype=torch.float32)

    # Create datasets and dataloaders for training and validation
    training_dataset = TensorDataset(training_data_tensor, training_labels_tensor)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = TensorDataset(validation_data_tensor, validation_labels_tensor)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    input_size = len(data[0])
    hidden_size = 15
    output_size = 1
    type_layer = None
    print('Running {} Classifier ...'.format(type_name))
    start = time.time()
    # Initialize the appropriate type of layer
    if type_name == 'rnn':
        type_layer = nn.RNN(input_size, hidden_size, batch_first=True)
    elif type_name == 'lstm':
        type_layer = nn.LSTM(input_size, hidden_size, batch_first=True)
    elif type_name == 'gru':
        type_layer = nn.GRU(input_size, hidden_size, batch_first=True)
    linear_layer = nn.Linear(hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(type_layer.parameters()) + list(linear_layer.parameters()), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        type_layer.train()
        for inputs, labels in training_dataloader:
            optimizer.zero_grad()
            outputs, _ = type_layer(inputs)
            outputs = linear_layer(outputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        type_layer.eval()  # Set the mode to evaluation
        with torch.no_grad():
            val_loss = 0
            for val_inputs, val_labels in validation_dataloader:
                val_outputs, _ = type_layer(val_inputs)
                val_outputs = linear_layer(val_outputs)
                val_loss += criterion(val_outputs.squeeze(), val_labels).item()
            val_loss /= len(validation_dataloader)

    with torch.no_grad():
        testing_outputs, _ = type_layer(testing_data_tensor)
        testing_outputs = linear_layer(testing_outputs)
        predictions = (testing_outputs.sigmoid() > 0.5).squeeze().numpy()  # Convert logits to binary predictions
        tf_sums = tf_pos_neg(predictions, testing_labels_tensor)
        accuracy, precision, recall, f1 = nn_eval(tf_sums[0], tf_sums[1], tf_sums[2], tf_sums[3])
    end = time.time()
    total = '{:.4f}'.format(end - start)
    print('Time to Run: {} seconds'.format(total))

    return accuracy, precision, recall, f1


def stacked_bilstm(data, data_labels, word2vec_model, epochs=50, batch_size=64, learning_rate=0.001):
    print('Running Bidirectional 3-layer Stacked LSTM Classifier ...')
    data, data_labels = tfx.comment_embeddings(data, data_labels, word2vec_model)

    training_data, testing_data, training_labels, testing_labels = (
        sklms.train_test_split(
            data,
            data_labels,
            test_size=0.2,
            stratify=data_labels,
            random_state=666
        )
    )
    validation_data, testing_data, validation_labels, testing_labels = sklms.train_test_split(
        testing_data,
        testing_labels,
        test_size=0.5,
        random_state=666
    )

    start = time.time()
    train_tensor = torch.tensor(training_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(training_labels, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_tensor = torch.tensor(validation_data, dtype=torch.float32)
    val_labels_tensor = torch.tensor(validation_labels, dtype=torch.float32)

    input_size = len(training_data[0])
    hidden_size = 15
    stacked_layers = 3
    output_size = 1

    # Create a bidirectional LSTM model
    lstm = nn.LSTM(input_size, hidden_size, num_layers=stacked_layers, batch_first=True, bidirectional=True)

    # Create a linear layer for the output
    linear = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(linear.parameters()), lr=learning_rate)

    best_val_loss = np.inf
    patience = 5
    early_stop_counter = 0

    # Training loop
    for epoch in range(epochs):
        train_loss = 0.0
        lstm.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # Initialize hidden state with zeros
            hidden_state = torch.zeros(stacked_layers * 2, inputs.size(0), hidden_size).to(inputs.device)
            cell_state = torch.zeros(stacked_layers * 2, inputs.size(0), hidden_size).to(inputs.device)
            outputs, _ = lstm(inputs)
            outputs = linear(outputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        with torch.no_grad():
            lstm.eval()
            val_out, _ = lstm(val_tensor)
            val_out = linear(val_out)
            val_loss = criterion(val_out.squeeze(), val_labels_tensor)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print("Early stopping at epoch", epoch)
                break

    # Evaluation on test data
    with torch.no_grad():
        test_tensor = torch.tensor(testing_data, dtype=torch.float32)
        test_labels_tensor = torch.tensor(testing_labels, dtype=torch.float32)
        # Initialize hidden state with zeros
        hidden_test = torch.zeros(stacked_layers * 2, test_tensor.size(0), hidden_size).to(test_tensor.device)
        cell_test = torch.zeros(stacked_layers * 2, test_tensor.size(0), hidden_size).to(test_tensor.device)

        # Forward propagate LSTM
        testing_outputs, _ = lstm(test_tensor)

        # Decode the hidden state of the last time step
        testing_outputs = linear(testing_outputs)
        test_predictions = (torch.sigmoid(testing_outputs) > 0.5).squeeze().numpy()
        true_positive_sum, false_positive_sum, false_negative_sum, true_negative_sum = tf_pos_neg(test_predictions,
                                                                                                  test_labels_tensor)
        accuracy, precision, recall, f1 = nn_eval(true_positive_sum, false_positive_sum, false_negative_sum,
                                                  true_negative_sum)
    end = time.time()
    total = '{:.4f}'.format(end - start)
    print('Time to Run: {} seconds'.format(total))

    return accuracy, precision, recall, f1
