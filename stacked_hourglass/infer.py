from predictor import HumanPosePredictor
from model import hg8

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import mpii_labels
import body25_labels
from analyze import calculate_angles, calculate_similarity_score, calculate_absolute_angle_scores

def draw_keypoints_on_image(image, keypoints, index=None):
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    joints = []
    for i in range(keypoints.shape[1]):
        joint = keypoints[0][i]
        joint_x = joint[0]
        joint_y = joint[1]
        if index is not None and index != i:
            continue
        plt.scatter(joint_x, joint_y, s=10, c='red', marker='o', label=i)
    plt.show()

def draw_skeleton_on_image(image, keypoints, target=plt, index=None):
    # fig,ax = plt.subplots(1)
    # ax.imshow(image)
    target.imshow(image)
    joints = []
    for i in range(keypoints.shape[1]):
        joint = keypoints[0][i]
        joint_x = joint[0]
        joint_y = joint[1]
        if index is not None and index != i:
            continue
        joints.append((joint_x, joint_y))
        target.scatter(joint_x, joint_y, s=10, c='red', marker='o', label=i)
    # draw skeleton
    for bone in mpii_labels.MPII_BONES:
        joint_1 = joints[bone[0]]
        joint_2 = joints[bone[1]]
        target.plot([joint_1[0], joint_2[0]], [joint_1[1], joint_2[1]], linewidth=3, alpha=0.7)
    # if target == plt:
    #     plt.show()

def draw_images(im1, im2, kp1, kp2, save, target_folder_path, frame_index):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle("Snapshot comparison")
    draw_skeleton_on_image(im1, kp1, ax1)
    draw_skeleton_on_image(im2, kp2, ax2)
    ax1.set_title("test")
    ax2.set_title("reference")
    if save:
        save_location = "../video_analysis/media/" + target_folder_path
        plt.savefig(save_location + "/frame{}.png".format(frame_index))
    plt.show()
    plt.close('all')
    


def compare_images(test_im_path, ref_im_path, save, target_folder_path="", i=0, debug=False, use_absolute_metric = False):
    model = hg8(pretrained=True)
    predictor = HumanPosePredictor(model, device='cpu')
    
    pil_test_img = Image.open(test_im_path).convert('RGB')
    pil_test_to_tensor = transforms.ToTensor()(pil_test_img).unsqueeze_(0)

    test_joints = predictor.estimate_joints(pil_test_to_tensor, flip=True)
    test_img = cv2.imread(test_im_path)

    pil_ref_img = Image.open(ref_im_path).convert('RGB')
    pil_ref_to_tensor = transforms.ToTensor()(pil_ref_img).unsqueeze_(0)

    ref_joints = predictor.estimate_joints(pil_ref_to_tensor, flip=True)
    ref_img = cv2.imread(ref_im_path)
    if save:
        draw_images(test_img, ref_img, test_joints.numpy(), ref_joints.numpy(), save, target_folder_path, i)    
    
    if use_absolute_metric:
        accuracy = calculate_absolute_angle_scores(ref_joints.numpy()[0], test_joints.numpy()[0], mpii_labels.MPII_BONES, bone_names=mpii_labels.MPII_BONES_NAMES, debug=debug)
    else:
        accuracy = calculate_similarity_score(ref_joints.numpy()[0], test_joints.numpy()[0], mpii_labels.MPII_ANGLES, mpii_labels.MPII_BONES, mpii_labels.MPII_ANGLES_NAMES)
    
    if debug:
        print("Test compared to reference accuracy: {}".format(accuracy))

    return accuracy
    
def main():
    acc = compare_images('./t1.jpg', './p1.jpg', False, debug=True, use_absolute_metric=False)
    print(acc)

if __name__ == "__main__":
    main()