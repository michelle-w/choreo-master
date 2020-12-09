from predictor import HumanPosePredictor
from model import hg8

from PIL import Image
from torchvision import transforms

# Get pretrained stacked hourglass model and returns estimated joint keypoints as numpy array. 
def get_keypoints(im_path):
    model = hg8(pretrained=True)
    predictor = HumanPosePredictor(model, device='cpu')
    
    pil_img = Image.open(im_path).convert('RGB')
    pil_tensor = transforms.ToTensor()(pil_img).unsqueeze_(0)

    joints = predictor.estimate_joints(pil_tensor, flip=True)
    return joints.numpy()

    
def main():
    pass
    # acc = compare_images('./t1.jpg', './p1.jpg', False, debug=True, use_absolute_metric=False)
    # print(acc)

if __name__ == "__main__":
    main()