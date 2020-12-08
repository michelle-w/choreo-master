from analyze import calculate_relative_angle_scores, calculate_absolute_angle_scores
from score_analysis import find_problem_times, find_top_n_problems
import body25_labels as b25

import numpy as np
import json
import os

def get_keypoints_from_json(frame_filename):
    with open(frame_filename) as f:
        data = json.load(f)
    if len(data['people']) == 0:
        print("No people found in ", frame_filename)
        return []
    all_data = data['people'][0]['pose_keypoints_2d']
    # remove the confidence scores
    kps_only = [i for j, i in enumerate(all_data) if (j+1)%3]
    # split into arrays for each keypoint
    split = len(all_data) / 3
    keypoints = np.array_split(kps_only, split)
    return keypoints

# TODO: add support for custom video names 
# Returns accuracies in 3 metrics: 1) relative angles without ranges of motion, 
# 2) relative angles with ranges of motion, 3) absolute angles to x-axis
def process_openpose_frames(name, folder_name1, folder_name2, num_frames, base_path = "./"):
    folder_path1, folder_path2 = os.path.join(base_path, folder_name1), os.path.join(base_path, folder_name2)

    num_frames1 = len(os.listdir(folder_path1))
    num_frames2 = len(os.listdir(folder_path2))
    num_frames = min(num_frames1, num_frames2, num_frames)

    rel_no_roms_accs = np.zeros((num_frames,))
    rel_roms_accs = np.zeros((num_frames,))
    abs_accs = np.zeros((num_frames,))

    prev_avg = 0

    for i in range(num_frames):
        number_str = str(i)
        zero_filled_number = number_str.zfill(12) # must have 12 digits
        f_name1 = "{}_test_{}_keypoints.json".format(name, zero_filled_number)

        number_str = str(i+shift)
        zero_filled_number = number_str.zfill(12) # must have 12 digits
        f_name2 = "{}_reference_{}_keypoints.json".format(name, zero_filled_number)
        
        f1_name = os.path.join(folder_path1, f_name1)
        f2_name = os.path.join(folder_path2, f_name2)
        
        kp1 = get_keypoints_from_json(f1_name)
        kp2 = get_keypoints_from_json(f2_name)

        if len(kp1) != 0 and len(kp2) != 0:
            rel_no_roms_acc = calculate_relative_angle_scores(kp1, kp2, b25.BODY25_ANGLE_LABELS, b25.BODY25_BONES, b25.BODY25_ANGLES_NAMES, b25.BODY25_ANGLE_ROMS, False)
            rel_roms_acc = calculate_relative_angle_scores(kp1, kp2, b25.BODY25_ANGLE_LABELS, b25.BODY25_BONES, b25.BODY25_ANGLES_NAMES, b25.BODY25_ANGLE_ROMS, True)
            abs_acc = calculate_absolute_angle_scores(kp1, kp2, b25.BODY25_ANGLE_LABELS, b25.BODY25_BONES)
            prev_avg = (rel_no_roms_acc + rel_roms_acc + abs_acc) / 3
        else:
            acc = prev_avg
        
        rel_no_roms_accs[i] = rel_no_roms_acc
        rel_roms_accs[i] = rel_roms_acc
        abs_accs[i] = abs_acc

    return rel_no_roms_accs, rel_roms_accs, abs_accs