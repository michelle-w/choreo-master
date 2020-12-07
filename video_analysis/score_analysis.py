#!/usr/.local/lib python
# -*- coding: utf-8 -*-


import numpy as np
import more_itertools as mit

# flag down sections with high amounts of differences (based on metrics, you’re doing this part of the dance wrong)
# e.g. from 0:02-0:05, the person is doing it completely wrong
# compute average score over 0.5 second increments (12 frames/half sec)
# set a threshold for percentage of losses that are too high
# combine contiguous increments that are above the threshold = section that is wrong
'''
combined_frame_num: frame index of comparison frames
fps: frames per second of video
offset: seconds that second video is ahead by

Returns

'''
def frame_to_secs(combined_frame_num, fps):
    secs_into_comparison = int(combined_frame_num * fps)
    return secs_into_comparison

def moving_average(x, w, s):
    all_avgs = np.convolve(x, np.ones(w), 'valid') / w
    return all_avgs[::s]

def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]



def find_problem_times(accs, fps, percentile = 25):
    threshold = np.percentile(accs, percentile)
    
    avgs = moving_average(accs, int(fps/2), int(fps/2))
    
    below_threshold = np.where(avgs < threshold, avgs, 0)
    problems = np.nonzero(below_threshold)

    print("Problems: ", problems)
    print("Problems type ", type(problems))

    durations = list(find_ranges(problems[0]))
    print("len durations ", len(durations))
    
    # avg -> second
    # problems = [1, 4, 9] -> [0.5, 2, 4.5]
    for i, duration in enumerate(durations):
        print("Dur in loop: ", duration)
        if type(duration) is tuple:
            durations[i] = (duration[0]/2, duration[1]/2)
        else:
            durations[i] = duration/2

    return durations


# provide the top n places for improvement 
# get top n local maximums in the score function (the worst n half second sections)
# return the timestamp + location (the angle/joints)
def find_top_n(accs, fps, n=10):
    avgs = moving_average(accs, int(fps/2), int(fps/2))
    
    top_n = np.argpartition((-1*avgs), -1*n)[-1*n:]

    top_n = top_n[np.argsort(avgs[top_n])]

    # index of avg * fps/2 = frame #
    top_n = np.multiply(top_n, int(fps/2))

    return top_n

def get_top_n_frames(top_n):
    for num in top_n:
        filename = "frame_{}".format(num)


def main():
    accs = np.load("aya_accs_offset.npy")
    fps = 24

    print(accs)

    top_five = find_top_n(accs, fps)
    print(top_five)

    problem_times = find_problem_times(accs, fps, percentile=25)
    print(problem_times)


if __name__ == "__main__":
    main()