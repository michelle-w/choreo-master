# from dataclasses import dataclass
from typing import List


# @dataclass
class DataInfo:
    def __init__(self, rgb_mean, rgb_stddev, joint_names, hflip_indices):
        self.rgb_mean = rgb_mean
        self.rgb_stddev = rgb_stddev
        self.joint_names = joint_names
        self.hflip_indices = hflip_indices
