import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

model = load_model("model_1.1_3.h5")

#inputdata = cv2.imread("predict.jpg", cv2.IMREAD_COLOR)
#image = cv2.resize(inputdata, (244, 244))
#image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
"""
image = cv2.imread("predict2.jpg", cv2.IMREAD_COLOR)

print(image.shape)

model_list = ["비타500", "쿨피스", "티즐"]

data = np.array([image])

prediction = model.predict(data)

print(prediction)

label = np.argmax(prediction)
"""
filename_train = ["vita500", "cool", "teazle"]

train_image = []
train_label = []

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

train_result = to_categorical(train_label)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1/255.0)
flow_train = train_gen.flow(train_image, train_result, batch_size=64, shuffle=False)

model.evaluate(flow_train)