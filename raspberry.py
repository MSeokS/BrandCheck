import RPi.GPIO as GPIO
from picamera import PiCamera
from gtts import gTTS
import pygame
import time
import predict

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

model = load_model("Resnet50_False_hyperband.h5")

with open("labels.txt", "r", encoding="utf-8") as file:
    labels = file.read().splitlines()

btn1 = 17
#btn2 = 2
#btn3 = 3
#rst = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(btn1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(btn2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(btn3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(rst, GPIO.IN, pull_up_down=GPIO.PUD_UP)

camera = PiCamera()

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

try:
    while 1:
        if GPIO.input(btn1):
            picture = take_picture()
            result = predict.predict()
            speak(labels[result])
            time.sleep(0.5)


#except GPIO.input(rst):
#    GPIO.cleanup

finally:
    GPIO.cleanup()
    camera.close()