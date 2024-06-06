from gpiozero import Button
from picamera2 import Picamera2
from adafruit_servokit import ServoKit
from gtts import gTTS
import pygame
import time
import predict
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

model = load_model("Resnet50_False_hyperband.h5")

with open("labels.txt", "r", encoding="utf-8") as file:
    labels = file.read().splitlines()
with open("prices.txt", "r", encoding="utf-8") as file:
    prices = file.read().splitlines()

kit = ServoKit(channels=12)
button_cam = Button(14)
button_add = Button(15)
button_rmv = Button(16)
button_rst = Button(17)


camera = Picamera2()
total_price = 0

def take_picture():
    image_path = "./predict.jpg"
    camera.capture(image_path)
    return image_path

def speak(text):
    tts = gTTS(text=text, lang='ko')
    tts.save("output.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(1)

    pygame.mixer.quit()
    
def braille(num, digit):
    if num == 0:
        kit.servo[digit + 0].angle = 135
        kit.servo[digit + 1].angle = 30
        kit.servo[digit + 2].angle = 135
        kit.servo[digit + 3].angle = 135
    elif num == 1:
        kit.servo[digit + 0].angle = 30
        kit.servo[digit + 1].angle = 135
        kit.servo[digit + 2].angle = 30
        kit.servo[digit + 3].angle = 30
    elif num == 2:
        kit.servo[digit + 0].angle = 30
        kit.servo[digit + 1].angle = 135
        kit.servo[digit + 2].angle = 30
        kit.servo[digit + 3].angle = 135
    elif num == 3:
        kit.servo[digit + 0].angle = 135
        kit.servo[digit + 1].angle = 135
        kit.servo[digit + 2].angle = 30
        kit.servo[digit + 3].angle = 30
    elif num == 4:
        kit.servo[digit + 0].angle = 135
        kit.servo[digit + 1].angle = 135
        kit.servo[digit + 2].angle = 135
        kit.servo[digit + 3].angle = 30
    elif num == 5:
        kit.servo[digit + 0].angle = 30
        kit.servo[digit + 1].angle = 135
        kit.servo[digit + 2].angle = 135
        kit.servo[digit + 3].angle = 30
    elif num == 6:
        kit.servo[digit + 0].angle = 135
        kit.servo[digit + 1].angle = 135
        kit.servo[digit + 2].angle = 30
        kit.servo[digit + 3].angle = 135
    elif num == 7:
        kit.servo[digit + 0].angle = 135
        kit.servo[digit + 1].angle = 135
        kit.servo[digit + 2].angle = 135
        kit.servo[digit + 3].angle = 135
    elif num == 8:
        kit.servo[digit + 0].angle = 30
        kit.servo[digit + 1].angle = 135
        kit.servo[digit + 2].angle = 135
        kit.servo[digit + 3].angle = 135
    else:
        kit.servo[digit + 0].angle = 135
        kit.servo[digit + 1].angle = 30
        kit.servo[digit + 2].angle = 30
        kit.servo[digit + 3].angle = 135    
        


"""
print("시작")
picture = take_picture()
img = predict.predict()
prediction = model.predict(img)
result = np.argmax(prediction)
speak(labels[result])
print(labels[result])

total_price = total_price + int(prices[result])
speak(prices[result] + "원")
print(prices[result])
mill = total_price // 10000
thou = (total_price % 10000) // 1000
hund = total_price % 1000
braille(mill, 0)
braille(thou, 4)
braille(hund, 8)
"""

while True:
    try:
        while True:
            print("시작")
            if button_cam.is_pressed:
                picture = take_picture()
                img = predict.predict()
                prediction = model.predict(img)
                result = np.argmax(prediction)
                speak(labels[result])
                print(labels[result])
                speak(prices[result] + "원")
                print(prices[result])
                while True:
                    if button_add.is_pressed:
                        total_price = total_price + int(prices[result])
                        mill = total_price // 10000
                        thou = (total_price % 10000) // 1000
                        hund = total_price % 1000
                        braille(mill, 0)
                        braille(thou, 4)
                        braille(hund, 8)
                        speak("구매 완료")
                        break
                    
                    if button_rmv.is_pressed:
                        speak("구매 취소")
                        break

    except button_rst.is_pressed:
        total_price = 0
        for i in range(12):
            kit.servo[i].angle = 30
        speak("초기화 완료")