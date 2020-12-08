#!/usr/.local/lib python
# -*- coding: utf-8 -*-

import numpy as np
import more_itertools as mit


# Helper function to calculate moving average of size w with stride s
def moving_average(x, w, s):
    all_avgs = np.convolve(x, np.ones(w), 'valid') / w
    return all_avgs[::s]

# Helper function to find ranges of consecutive numbers
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]

# Flag sections with high amounts of differences, i.e. scores above a certain threshold.
# Default threshold is based on 25th percentile of scores.
def find_problem_times(accs, fps, percentile = 25):
    # Set a threshold for percentage of losses that are too high
    threshold = np.percentile(accs, percentile)
    
    # Average scores over 0.5 second increments (e.g. for 24fps, 12 frames/half sec)
    avgs = moving_average(accs, int(fps/2), int(fps/2))
    
    # Combine contiguous increments that are above the threshold = sections that are wrong
    below_threshold = np.where(avgs < threshold, avgs, 0)
    problems = np.nonzero(below_threshold)
    sections = list(find_ranges(problems[0]))
    
    # Convert average index to seconds in video.
    # e.g. problems = [1, 4, 9] -> [0.5, 2, 4.5]
    for i, section in enumerate(sections):
        if type(section) is tuple:
            sections[i] = (section[0]/2, section[1]/2)
        else:
            sections[i] = (section/2, section/2 + 0.5)

    return sections


# Returns a list of the top n places for improvement in the form of starting 0.5 second section timestamp. 
# By default, the function will return the top 5 half second sections that had the lowest scores.
def find_top_n_problems(accs, fps, n=5):
    avgs = moving_average(accs, int(fps/2), int(fps/2))

    # Get top n local minimums in the score function (the worst n half second sections).
    top_n = np.argpartition((-1*avgs), -1*n)[-1*n:]

    # Get the indices of the minimums in descending order.
    top_n = top_n[np.argsort(-1*avgs[top_n])]

    # Convert index to timestamp, i.e. index of min / 2.
    top_n = np.divide(top_n, 2)

    return top_n

# Gets the corresponding frame for the given timestamp.
def get_image_with_timestamp(timestamp):
    # Convert index to frame number, i.e. timestamp * fps.
    frame_number = timestamp*fps

    filename = "frame_{}".format(frame_number)

# Generate feedback report with the problem sections in the 25th percentile and top 10 sections for improvement.
def generate_feedback_report(name, accs, fps):
    report = []
    report.append("Report for {}".format(name))
    report.append("------------------------------\n")

    problem_times = find_problem_times(accs, fps)
    report.append("Here are the sections that were most different from the reference:")
    for time in problem_times:
        report.append("> {} - {} seconds".format(time[0], time[1]))

    report.append("\n----------------------------\n")

    top_5_times = find_top_n_problems(accs, fps)
    report.append("Here are the top 5 most different 0.5 second sections:")
    for time in top_5_times:
        report.append("> {} - {} seconds".format(time, time+0.5))

    report.append("\n------------------------------")

    return ("\n".join(report))


def main():
    accs = np.load("aya_accs_offset.npy")
    fps = 24

    print(accs)

    top_five = find_top_n_problems(accs, fps)
    print(top_five)

    problem_times = find_problem_times(accs, fps, percentile=25)
    print(problem_times)


if __name__ == "__main__":
    main()