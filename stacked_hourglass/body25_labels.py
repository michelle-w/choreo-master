HEAD = 0
CHEST = 1
L_SHOULDER = 2
L_ELBOW = 3
L_WRIST = 4
R_SHOULDER = 5
R_ELBOW = 6
R_WRIST = 7
PELVIS = 8
R_HIP = 9
R_KNEE = 10
R_ANKLE = 11
L_HIP = 12
L_KNEE = 13
L_ANKLE = 14
R_FOOT = 19
L_FOOT = 22

BODY25_BONES = [
    [R_KNEE, R_ANKLE],
    [R_HIP, R_KNEE],
    [R_HIP, PELVIS],
    [L_HIP, PELVIS],
    [L_HIP, L_KNEE],
    [L_KNEE, L_ANKLE],
    [PELVIS, CHEST],
    [CHEST, HEAD],
    [R_WRIST, R_ELBOW],
    [R_ELBOW, R_SHOULDER],
    [CHEST, R_SHOULDER],
    [CHEST, L_SHOULDER],
    [L_SHOULDER, L_ELBOW],
    [L_ELBOW, L_WRIST],
    [R_ANKLE, R_FOOT],
    [L_ANKLE, L_FOOT]
]

RIGHT_LOWER_LEG = 0
RIGHT_UPPER_LEG = 1
RIGHT_LEG_JOINT = 2
LEFT_LEG_JOINT = 3
LEFT_UPPER_LEG = 4
LEFT_LOWER_LEG = 5
SPINE = 6
HEAD_TO_CHEST = 7
RIGHT_LOWER_ARM = 8
RIGHT_UPPER_ARM = 9
RIGHT_SHOULDER_JOINT = 10
LEFT_SHOULDER_JOINT = 11
LEFT_UPPER_ARM = 12
LEFT_LOWER_ARM = 13
RIGHT_FOOT = 14
LEFT_FOOT = 15

BODY25_ANGLES = [
    [RIGHT_LOWER_LEG, RIGHT_UPPER_LEG],
    [LEFT_LOWER_LEG, LEFT_UPPER_LEG],
    [LEFT_LEG_JOINT, LEFT_UPPER_LEG], # less important
    [RIGHT_LEG_JOINT, RIGHT_UPPER_LEG], # less important
    [SPINE, HEAD_TO_CHEST],
    [RIGHT_LOWER_ARM, RIGHT_UPPER_ARM],
    [RIGHT_SHOULDER_JOINT, RIGHT_UPPER_ARM],
    [RIGHT_SHOULDER_JOINT, HEAD_TO_CHEST], # less important
    [LEFT_LOWER_ARM, LEFT_UPPER_ARM],
    [LEFT_SHOULDER_JOINT, LEFT_UPPER_ARM],
    [LEFT_SHOULDER_JOINT, HEAD_TO_CHEST], # less important
    [RIGHT_FOOT, RIGHT_LOWER_LEG],
    [LEFT_FOOT, LEFT_LOWER_LEG]
]

BODY25_ANGLES_NAMES = [
    "[RIGHT_LOWER_LEG, RIGHT_UPPER_LEG]",
    "[LEFT_LOWER_LEG, LEFT_UPPER_LEG]",
    "[LEFT_LEG_JOINT, LEFT_UPPER_LEG]", # less important
    "[RIGHT_LEG_JOINT, RIGHT_UPPER_LEG]", # less important
    "[SPINE, HEAD_TO_CHEST]", # possibly take out later
    "[RIGHT_LOWER_ARM, RIGHT_UPPER_ARM]",
    "[RIGHT_SHOULDER_JOINT, RIGHT_UPPER_ARM]",
    "[RIGHT_SHOULDER_JOINT, HEAD_TO_NECK]", # less important
    "[LEFT_LOWER_ARM, LEFT_UPPER_ARM]",
    "[LEFT_SHOULDER_JOINT, LEFT_UPPER_ARM]",
    "[LEFT_SHOULDER_JOINT, HEAD_TO_NECK]", # less important
    "[RIGHT_FOOT, RIGHT_LOWER_LEG]",
    "[LEFT_FOOT, LEFT_LOWER_LEG]"
]
