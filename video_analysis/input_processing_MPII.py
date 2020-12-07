#!/usr/local/bin python
# -*- coding: utf-8 -*-

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
sys.path.append("../stacked_hourglass")
from stacked_hourglass.infer import *
from stacked_hourglass.analyze import *
import datetime

'''
Parameters
file_name:  string name of file not including extension
file_type:  extension of file type
start_time: time into the video to start in ms
end_time:   time into the video to stop saving in ms
target_folder_name: folder name that frames should be saved into
base_path:  path of where videos are and frames should be saved
max_frames: max number of frames that we want to save
'''
def extract_frames(file_name, file_type, start_time, end_time, desired_fps = 24, target_folder_name = "frames", base_path = "./", max_frames = 10000):
    if target_folder_name == "frames":
        target_folder_name = file_name + "_frames"
    target_folder_name = os.path.join(base_path, target_folder_name)
    cam = cv2.VideoCapture(os.path.join(base_path, file_name + "." + file_type))
    curr_ms = start_time
    timestep = 1000/desired_fps
  
    # save extracted frames to a folder
    try:  
        # creating a folder
        if not os.path.exists(target_folder_name): 
            os.makedirs(target_folder_name)

    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 

    # frame 
    currentframe = 0
    saved_count = 0

    while(True): 
        # reading from frame 
        cam.set(cv2.CAP_PROP_POS_MSEC, curr_ms)
        ret, frame = cam.read() 
        
        # print("relative position: ", cam.get(cv2.CAP_PROP_POS_AVI_RATIO ))
        # print("frame num: ", cam.get(cv2.CAP_PROP_POS_FRAMES ))

        if currentframe > max_frames:
            break

        currentframe += 1
    
        if curr_ms <= end_time and ret: # we want to be saving and have video remaining
            # if video is still left continue creating images 
            # name = './data/frame' + str(currentframe) + '.jpg'
            # print ('Creating...' + name) 
            # name = os.path.join(target_folder_name, "frame{}_{1:.{2}f}ms_.png".format(saved_count, curr_ms, MS_ROUNDING))

            name = os.path.join(target_folder_name, "frame{}.png".format(saved_count))
            # writing the extracted images 
            cv2.imwrite(name, frame) 
    
            # increasing counter so that it will 
            # show how many frames are created 
            saved_count += 1

            # increment curr_ms by time interval
            curr_ms += timestep
            
        else:
            break

    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

def process_frames(folder_name1, folder_name2, num_frames, target_folder_name = "", base_path = "./", offset = 0, save = True):
    folder_path1, folder_path2 = os.path.join(base_path, folder_name1), os.path.join(base_path, folder_name2)
    if target_folder_name == "":
        target_folder_name = "{}_vs_{}".format(folder_name1, folder_name2)
    target_folder_path = os.path.join(base_path, target_folder_name)

    # save extracted frames to a folder
    try:  
        # creating a folder
        if not os.path.exists(target_folder_path): 
            os.makedirs(target_folder_path)

    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 


    num_frames1 = len(os.listdir(folder_path1))
    num_frames2 = len(os.listdir(folder_path2))
    shift = int(offset * 24)

    num_frames = min(num_frames1, num_frames2, num_frames)
    accs = np.zeros((num_frames,))

    for i in range(num_frames):
        f_name1 = "frame{}.png".format(i)
        f_name2 = "frame{}.png".format(i+shift)
        f1_name = os.path.join(folder_path1, f_name1)
        f2_name = os.path.join(folder_path2, f_name2)
        acc = compare_images(f1_name, f2_name, save, target_folder_name, i)
        # plt.show()
        # plt.savefig(os.path.join(target_folder_path, f_name))
        accs[i] = acc
    return accs

def create_end_video(frame_folder_path, target_file_path, fps, num_frames):
    i = 0
    img = cv2.imread(frame_folder_path + "/frame{}.png".format(i))
    height, width, layers = img.shape
    size = (width,height)
    out = cv2.VideoWriter(target_file_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(num_frames):
        img = cv2.imread(frame_folder_path + "/frame{}.png".format(i))
        out.write(img)
    
    out.release() 

def test_write(frame_folder_path, target_file_path):
    frame_folder_path = "./media/aya_comparison_offset"
    target_file_path = "./media/aya_comparison_offset_video_short.mp4"
    fps = 24
    num_frames = 150
    create_end_video(frame_folder_path, target_file_path, fps, num_frames)

def main():
    start_time = datetime.datetime.now()
    offset = 7.97
    print(start_time)
    accs = process_frames(folder_name1 = "aya_test_frames", folder_name2 = "aya_ref_frames", num_frames = 500, target_folder_name = "aya_comparison_offset_gcp", base_path = "./media", offset = offset, save=True)
    np.save("aya_accs_offset_gcp.npy", accs)
    test_write("./media/aya_comparison_offset_gcp", "./media/aya_comparison_offset_video_gcp.mp4")
    end_time = (datetime.datetime.now())
    print(end_time)

if __name__ == "__main__":
    main()
