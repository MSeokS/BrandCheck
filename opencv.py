import cv2


# 프레임 추출
def extract_frames(video_file):
    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    frames = []

    while success:
        frames.append(image)
        success, image = vidcap.read()

    # 동영상의 전체 프레임 수 확인
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames:', total_frames)

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

video = "vita500.mp4"
frames_list = extract_frames(video)
resized_images = resize_images(frames_list, 244, 244) 

# 이미지 확인용
for idx, resized_image in enumerate(resized_images):
    cv2.imshow(f"Resized Image {idx}", resized_image)
    print(resized_image.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()