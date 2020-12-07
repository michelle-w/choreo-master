from analyze import calculate_similarity_score, calculate_absolute_angle_scores
from score_analysis import find_problem_times, find_top_n
import body25_labels

def get_keypoints_from_json(filename):
    with open(filename) as f:
        data = json.load(f)
    if len(data['people']) == 0:
        print("No people found in ", filename)
        return []
    all_data = data['people'][0]['pose_keypoints_2d']
    # remove the confidence scores
    kps_only = [i for j, i in enumerate(all_data) if (j+1)%3]
    # split into arrays for each keypoint
    split = len(all_data) / 3
    keypoints = np.array_split(kps_only, split)
    return keypoints

def process_frames(folder_name1, folder_name2, num_frames, target_folder_name = "", base_path = "./", offset = 0, save = True, use_absolute_metric = False):
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

    avg_so_far = 0

    for i in range(num_frames):
        number_str = str(i)
        zero_filled_number = number_str.zfill(3) # 3 leading digits
        f_name1 = "test_bad_000000000{}_keypoints.json".format(zero_filled_number)

        number_str = str(i+shift)
        zero_filled_number = number_str.zfill(3) # 3 leading digits
        f_name2 = "reference_000000000{}_keypoints.json".format(zero_filled_number)
        
        f1_name = os.path.join(folder_path1, f_name1)
        f2_name = os.path.join(folder_path2, f_name2)
        
        kp1 = get_keypoints_from_json(f1_name)
        kp2 = get_keypoints_from_json(f2_name)

        if len(kp1) != 0 and len(kp2) != 0:
            if use_absolute_metric:
                accuracy = calculate_absolute_angle_scores(ref_joints.numpy()[0], test_joints.numpy()[0], mpii_labels.MPII_BONES, bone_names=mpii_labels.MPII_BONES_NAMES, debug=debug)
            else:
                accuracy = calculate_similarity_score(ref_joints.numpy()[0], test_joints.numpy()[0], mpii_labels.MPII_ANGLES, mpii_labels.MPII_BONES, mpii_labels.MPII_ANGLES_NAMES)
                avg_so_far = (i*avg_so_far + acc) / (i + 1)
        else:
          acc = avg_so_far
        
        # plt.show()
        # plt.savefig(os.path.join(target_folder_path, f_name))

        accs[i] = acc
    return accs