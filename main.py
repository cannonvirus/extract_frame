import os
import numpy as np
import cv2
import yaml

def main(config):
    print("----------------------------")
    print("Templete Python Project")
    print("----------------------------")

    cap = cv2.VideoCapture(config['video_path'])
    frame_count = 0

    while(1):
        ret, frame = cap.read() 

        if not ret:
            break
        
        if frame_count % config['save_frame_count'] == 0:
            print("frame : {}".format(frame_count))
            cv2.imwrite(  os.path.join(config['save_img_path'], "{}.jpg".format(str(frame_count).zfill(5))), frame )

        frame_count += 1


if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    main(config)


