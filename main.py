import os
import numpy as np
import cv2
import yaml
import albumentations as A

def main(config):
    print("----------------------------")
    print("Templete Python Project")
    print("----------------------------")

    cap = cv2.VideoCapture(config['video_path'])
    first_name = os.path.splitext(os.path.basename(config['video_path']))[0]

    # if not os.path.isdir(os.path.join(config['out_path'], first_name)):
    #     os.mkdir(os.path.join(config['out_path'], first_name))

    frame_count = 0

    while(1):
        ret, frame = cap.read() 

        if not ret:
            break
        
        # X : 240 ~ 1680
        if frame_count % config['save_frame_count'] == 0:

            if not os.path.isdir(os.path.join(config['out_path'], "1_" + first_name + "_" +  str(frame_count).zfill(5))):
                os.mkdir(os.path.join(config['out_path'], "1_" + first_name + "_" +  str(frame_count).zfill(5)))

            clean_img = frame[:,240:1680,:]
            for id in range(10):
                transform = A.Compose([A.RandomSizedCrop(min_max_height=[500,1000], height=config['out_img_size'], width=config['out_img_size'], p=1)])
                transformed = transform(image=clean_img)
                cv2.imwrite(os.path.join( config['out_path'], "1_" + first_name + "_" +  str(frame_count).zfill(5), "{}.jpg".format(str(id).zfill(3))), transformed["image"])

            frame_re = cv2.resize(frame, (config['out_img_size'], config['out_img_size']))
            cv2.imwrite(os.path.join(config['out_path'], "1_" + first_name + "_" +  str(frame_count).zfill(5), "origin.jpg"), frame_re)

            print("frame : {}".format(frame_count))

        frame_count += 1


if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    main(config)


