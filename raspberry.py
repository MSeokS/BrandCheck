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
with open("events.txt", "r", encoding="utf-8") as file:
    events = file.read().splitlines()

kit = ServoKit(channels=16)
button_cam = Button(15)
button_add = Button(18)
button_rmv = Button(17)
button_rst = Button(14)


picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.start()

total_price = 0

def take_picture():
    image_path = "./predict.jpg"
    picam2.capture_file(image_path)
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


print("시작")
try:
    while True:
        if button_cam.is_pressed:
            picture = take_picture()
            img = predict.predict()
            prediction = model.predict(img)
            result = np.argmax(prediction)
            speak(labels[result])
            print(labels[result])
            speak(prices[result] + "원")
            print(prices[result])
            if events[result] == "1":
                print("원플러스원")
                speak("원플러스원 행사상품입니다.")
            elif events[result] == "2":
                print("투플러스원")
                speak("투플러스원 행사상품입니다.")
            while True:
                if button_add.is_pressed:
                    total_price = total_price + int(prices[result])
                    mill = total_price // 10000
                    thou = (total_price % 10000) // 1000
                    hund = total_price % 1000
                    braille(mill, 0)
                    braille(thou, 4)
                    braille(hund, 8)
                    print("구매 완료")
                    speak("구매 완료")
                    print("total price : %d", total_price)
                    break
                    
                if button_rmv.is_pressed:
                    print("구매 취소")
                    speak("구매 취소")
                    print("total price : %d", total_price)
                    break

                if button_rst.is_pressed:
                    total_price = 0
                    for i in range(12):
                        kit.servo[i].angle = 30
                    print("초기화 완료")
                    speak("초기화 완료")
                    print("total price : %d", total_price)
                    break

        if button_rst.is_pressed:
                total_price = 0
                for i in range(12):
                    kit.servo[i].angle = 30
                print("초기화 완료")
                speak("초기화 완료")
                print("total price : %d", total_price)

except KeyboardInterrupt:
    print("프로그램 종료")
