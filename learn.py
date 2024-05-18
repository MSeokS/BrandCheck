import os
import tensorflow as tf
import IPython
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
def model_builder(hp):
    model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    output = model.output

    x = GlobalAveragePooling2D()(output)
    hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
    x = Dense(units = hp_units, activation='relu')(x)
    output = Dense(100, activation='softmax', name='output')(x)

    model = Model(inputs=model.input, outputs=output)

    hp_lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#라벨 제작

import numpy as np

def transform_images(x_train):
        x_train = tf.image.resize(x_train, (256, 256))
        x_train = tf.image.random_crop(x_train, (224, 224, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train

def parse_tfrecord(tfrecord):
    features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                'image/filename': tf.io.FixedLenFeature([], tf.string),
                'image/encoded': tf.io.FixedLenFeature([], tf.string)}
    x = tf.io.parse_single_example(tfrecord, features)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = transform_images(x_train)
    y_train = x['image/source_id']
    y_train = tf.one_hot(y_train, 100)
    return x_train, y_train

def parse_tfrecord_valid(tfrecord):
    features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                'image/filename': tf.io.FixedLenFeature([], tf.string),
                'image/encoded': tf.io.FixedLenFeature([], tf.string)}
    x = tf.io.parse_single_example(tfrecord, features)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (224, 224)
    y_train = x['image/source_id']
    y_train = tf.one_hot(y_train, 100)
    return x_train, y_train


train_data = tf.data.TFRecordDataset('train.tfrecord')
train_data = train_data.map(parse_tfrecord)

valid_data = tf.data.TFRecordDataset('valid.tfrecord')
valid_data = valid_data.map(parse_tfrecord_valid)

#train_data = train_data.shuffle(buffer_size=10000)
train_data = train_data.batch(64)
train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

valid_data = valid_data.batch(64)
valid_data = valid_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#라벨 제작 끝

# train_image, val_image, train_result, val_result

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import kerastuner as kt
# model.compile(optimizer=Adam(0.048), loss='categorical_crossentropy', metrics=['accuracy'])

tuner = kt.Hyperband(model_builder,
        objective = 'val_accuracy',
        max_epochs = 10,
        factor = 3,
        directory = './log/', 
        project_name = 'kt_test')

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

tuner.search(train_data, epochs=10, validation_data=valid_data, callbacks=[ClearTrainingOutput()])
best = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"{best.get('units')} {best.get('learning_rate')}");

model = tuner.hypermodel.build(best)
result = model.fit(train_data, epochs=50, validation_data=valid_data)

#model.evaluate(flow_test)

from tensorflow.keras.models import save_model
model.save("Resnet50_False_hyperband.h5")
