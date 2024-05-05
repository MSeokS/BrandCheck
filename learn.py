import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

model = ResNet50(input_shape=(244, 244, 3), include_top=False, weights='imagenet')
output = model.output

x = GlobalAveragePooling2D()(output)
x = Dense(64, activation='relu')(x)
output = Dense(3, activation='softmax', name='output')(x)

model = Model(inputs=model.input, outputs=output)

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#라벨 제작

import numpy as np
import os
import cv2

filename_train = ["vita500", "cool", "teazle"]
filename_test = []

train_image = []
tr_im = []
train_label = []
test_image = []
test_label = []

for i in range(len(filename_train)):
    image_dir = f"./{filename_train[i]}/"
    image_files = [os.path.join(image_dir, name) for name in os.listdir(image_dir) if name.endswith(('jpg'))]

    for file_path in image_files:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is not None:
            train_image.append(image)
            train_label.append(i)

train_image = np.array(train_image)
train_label = np.array(train_label)

print(train_label)

for i in range(len(filename_test)):
    image_dir = f"./{filename_test[i]}/"
    image_files = [os.path.join(image_dir, name) for name in os.listdir(image_dir) if name.endswith(('jpg'))]

    for file_path in image_files:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is not None:
            test_image.append(image)
            test_label.append(i)

test_image = np.array(test_image)
test_label = np.array(test_label)

train_result = to_categorical(train_label)
#test_result = to_categorical(test_label)
train_image, val_image, train_result, val_result = train_test_split(train_image, train_result, test_size = 0.15)

print(train_image.shape, train_result.shape)
print(val_image.shape, val_result.shape)
print(val_result)

#라벨 제작 끝

# train_image, val_image, train_result, val_result

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

train_gen = ImageDataGenerator(
        rotation_range=90,
        rescale=1/255.0
)

val_gen = ImageDataGenerator(rescale=1/255.0)
test_gen = ImageDataGenerator(rescale=1/255.0)

flow_train = train_gen.flow(train_image, train_result, batch_size=64, shuffle=True)
flow_val = val_gen.flow(val_image, val_result, batch_size=64, shuffle=False)
#flow_test = test_gen.flow(test_image, test_result, batch_size=64, shuffle=False)

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])


reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, mode='min', verbose=1)
stop_cb = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

result = model.fit(flow_train, epochs=50, validation_data=flow_val, callbacks=[reduce_lr_cb, stop_cb])

#model.evaluate(flow_test)

from tensorflow.keras.models import save_model
model.save("model_1.1_3.h5")