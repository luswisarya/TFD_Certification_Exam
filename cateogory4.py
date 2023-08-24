# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    with open('sarcasm.json', 'r') as json_read:
        getdata = json.load(json_read)

    for i in getdata:
        sentences.append(i['headline'])
        labels.append(i['is_sarcastic'])

    train_sentences = sentences[:training_size]
    train_labels = np.array(labels[:training_size])
    test_sentences = sentences[training_size:]
    test_labels = np.array(labels[training_size:])

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size,
                          oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)

    training_seq = tokenizer.texts_to_sequences(train_sentences)
    train_sentences_pad = pad_sequences(training_seq,
                                        maxlen=max_length,
                                        truncating=trunc_type,
                                        padding=padding_type)

    testing_seq = tokenizer.texts_to_sequences(test_sentences)
    test_sentences_pad = pad_sequences(testing_seq,
                                       maxlen=max_length,
                                       truncating=trunc_type,
                                       padding=padding_type)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callback
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            # Check accuracy
            if (logs.get('accuracy') > 0.90 and logs.get('val_accuracy') > 0.75):
                # Stop if threshold is met
                print("\n\n\t=====================================")
                print("\t|| accuracy and val_accuracy > 95% ||")
                print("\t=====================================\n")
                self.model.stop_training = True


    callbacks = myCallback()

    model.fit(train_sentences_pad,
              train_labels,
              epochs=20,
              validation_data=(test_sentences_pad, test_labels),
              callbacks=[callbacks]
              )
    return model




# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
