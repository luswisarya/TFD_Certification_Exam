# ==============================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative to its
# difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# WARNING: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure. You do not need them to solve the question.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ==============================================================================
#
# COMPUTER VISION WITH CNNs
#
# Create and train a classifier to classify images between three categories
# of beans using the beans dataset.
# ==============================================================================
# ABOUT THE DATASET
#
# Beans dataset has images belonging to 3 classes as follows:
# 2 disease classes (Angular leaf spot, bean rust)
# 1 healthy class (healthy).
# The images are of different sizes and have 3 channels.
# ==============================================================================
#
# INSTRUCTIONS
#
# We have already divided the data for training and validation.
#
# Complete the code in following functions:
# 1. preprocess()
# 2. solution_model()
#
# Your code will fail to be graded if the following criteria are not met:
# 1. The input shape of your model must be (300,300,3), because the testing
#    infrastructure expects inputs according to this specification. You must
#    resize all the images in the dataset to this size while pre-processing
#    the dataset.
# 2. The last layer of your model must be a Dense layer with 3 neurons
#    activated by softmax since this dataset has 3 classes.
#
# HINT: Your neural network must have a validation accuracy of approximately
# 0.75 or above on the normalized validation dataset for top marks.
#


import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Use this constant wherever necessary
IMG_SIZE = 300

# This function normalizes and resizes the images.

# COMPLETE THE CODE IN THIS FUNCTION
def preprocess(image, label):
    # RESIZE YOUR IMAGES HERE (HINT: After resizing the shape of the images
    # should be (300, 300, 3).
    image = tf.image.resize(image, (300, 300))
    # NORMALIZE YOUR IMAGES HERE (HINT: Rescale by 1/.255))
    image = image / 255.0

    return image, label


# This function loads the data, normalizes and resizes the images, splits it into
# train and validation sets, defines the model, compiles it and finally
# trains the model. The trained model is returned from this function.

# COMPLETE THE CODE IN THIS FUNCTION.
def solution_model():
    # Loads and splits the data into training and validation splits using tfds.
    (ds_train, ds_validation), ds_info = tfds.load(
        name='beans',
        split=['train', 'validation'],
        as_supervised=True,
        with_info=True)

    BATCH_SIZE = 32

    # Resizes and normalizes train and validation datasets using the
    # preprocess() function.
    # Also makes other calls, as evident from the code, to prepare them for
    # training.
    ds_train = ds_train.map(preprocess).cache().shuffle(
        ds_info.splits['train'].num_examples).batch(BATCH_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE)
    ds_validation = ds_validation.map(preprocess).batch(BATCH_SIZE).cache(

    ).prefetch(tf.data.experimental.AUTOTUNE)

    # Custom Callback
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > 0.75 and logs.get('val_accuracy') > 0.75):
                print("\nAccuracy is more than 75%, stopping...")
                self.model.stop_training = True

    customCallback = myCallback()

    # Code to define the model
    model = tf.keras.models.Sequential([

        # ADD LAYERS OF THE MODEL HERE
        Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        # If you don't adhere to the instructions in the following comments,
        # tests will fail to grade your model:
        # The input layer of your model must have an input shape of
        # (300,300,3).
        # Make sure that your last layer has 3 (number of classes) neurons
        # activated by softmax.
        tf.keras.layers.Dense(3, activation='softmax'),
    ])

    # Code to compile and train the model
    model.compile(
        # YOUR CODE HERE
        optimizer=RMSprop(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])

    model.fit(
        # YOUR CODE HERE
        ds_train,
        epochs=10,
        verbose=1,
        validation_data=ds_validation,
        callbacks=[customCallback]
    )
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
