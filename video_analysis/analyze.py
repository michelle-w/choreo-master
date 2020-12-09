import numpy as np

def calculate_angles(keypoints, angle_labels, bone_labels, angle_name_labels, debug=False):
    angles = []
    for i, bones in enumerate(angle_labels):
        k1 = keypoints[bone_labels[bones[0]][0]]
        k2 = keypoints[bone_labels[bones[0]][1]]
        k3 = keypoints[bone_labels[bones[1]][0]]
        k4 = keypoints[bone_labels[bones[1]][1]]
        intersection = k1 if (np.equal(k1, k3).all() or np.equal(k1, k4).all()) else k2
        if np.equal(intersection, k1).all():
            vec1 = intersection - k2
        else:
            vec1 = intersection - k1
        
        if np.equal(intersection, k3).all():
            vec2 = intersection - k4
        else:
            vec2 = intersection - k3
        a1 = np.arctan2(vec1[1], vec1[0])
        a2 = np.arctan2(vec2[1], vec2[0])
        angle = a2 - a1
        if (angle < 0):
            angle += 2 * np.pi
        angles.append(np.degrees(angle))

    for i in range(len(angles)):
        bones = angle_labels[i]
        vec1 = np.array(keypoints[bone_labels[bones[0]][0]] - keypoints[bone_labels[bones[0]][1]])
        vec2 = np.array(keypoints[bone_labels[bones[1]][0]] - keypoints[bone_labels[bones[1]][1]])
        if debug:
            print("{} : {}, vec1 = {}, vec2 = {}".format(angle_name_labels[i], angles[i], vec1, vec2))
    
    return angles

def calculate_relative_angle_scores(kp1, kp2, angle_labels, bone_labels, angle_name_labels, angle_roms, with_ranges_of_motion, debug=False):
    a1 = calculate_angles(kp1, angle_labels, bone_labels, angle_name_labels)
    a2 = calculate_angles(kp2, angle_labels, bone_labels, angle_name_labels)

    sum_diffs = 0
    for i, a in enumerate(angle_labels):
        diff = abs(a1[i] - a2[i])
        if with_ranges_of_motion:
            diff /= (angle_roms[i] * 2)
        sum_diffs += diff
        if debug:
            print("For angle {}, difference is {}".format(angle_name_labels[i], a1[i] - a2[i]))
    score = (sum_diffs / len(angle_labels))
    if not with_ranges_of_motion:
        score /= 180
    return 1 - score

    
def calculate_absolute_angle_scores(kp1, kp2, bone_labels, bone_names=None, debug=False):
    sum_diffs = 0
    bone_count = 0
    for i, b in enumerate(bone_labels):
        v1 = kp1[b[0]] - kp1[b[1]]
        v2 = kp2[b[0]] - kp2[b[1]]
        # skip the keypoint if it wasn't detected
        if v1[1] == 0 or v2[1] == 0:
            continue
        bone_count += 1
        a1 = np.degrees(np.arctan2(v1[1], v1[0]))
        a2 = np.degrees(np.arctan2(v2[1], v2[0]))
        if (a1 < 0):
            a1 += 360
        if (a2 < 0):
            a2 += 360
        diff = a1 - a2
        sum_diffs += abs(diff)
        if debug:
            print("For bone {}:\ntest angle: {},\nref angle: {},\ndifference is {}".format(bone_names[i], a1, a2, diff))
        
    return 1 - ((sum_diffs / bone_count) / 180)

