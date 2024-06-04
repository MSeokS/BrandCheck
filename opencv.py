import cv2
import os

def extract_frames(video_file):
    vidcap = cv2.VideoCapture(video_file)

    # 동영상의 전체 프레임 수 확인
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames:', total_frames)

    # 프레임 간격 설정
    frame_interval = total_frames // 240  # 240개 프레임을 추출하도록 간격 설정
    print('Frame interval:', frame_interval)

    frames = []
    frame_idx = 0

    while True:
        success, image = vidcap.read()
        if not success:
            break

        # 매 frame_interval 번째 프레임 저장
        if frame_idx % frame_interval == 0:
            frames.append(image)

        frame_idx += 1

    # 동영상의 프레임 크기 확인
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Frame size:', width, 'x', height)

    vidcap.release()
    return frames

#이미지 크기 조정
def resize_images(image_list, width, height):
    resized_images = []
    for image in image_list:
        resized_image = cv2.resize(image, (width, height))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_images.append(rgb_image)
    return resized_images

video_list = ["test"]

for drink in video_list:
    video = drink + ".mp4"
    frames_list = extract_frames(video)
    resized_images = resize_images(frames_list, 244, 244) 

    image_path = f"./{drink}/"

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i in range(len(resized_images)):
        cv2.imwrite(image_path + f"{drink}_{i}.jpg", resized_images[i])