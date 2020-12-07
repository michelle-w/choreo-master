from moviepy.editor import *

import align_videos_by_soundtrack.align as align
import align_videos_by_soundtrack.cli_common as cli_common
from align_videos_by_soundtrack.utils import check_and_decode_filenames

# Dependent on reference video being longer than the test video
def calculate_offset(test_video_filename, ref_video_filename):
    file_specs = [test_video_filename, ref_video_filename]
    with align.SyncDetector() as det:
        result = det.align(file_specs)

    report = []
    amount = 0
    for i, path in enumerate(file_specs):
        if not (result[i]["trim"] > 0):
            continue
#         report.append(
#             """Result: The beginning of '%s' needs to be trimmed off %.4f seconds \
# (or to be added %.4f seconds padding) for all files to be in sync""" % (
#                 path, result[i]["trim"], result[i]["pad"]))
        amount = result[i]["trim"]

    return round(amount, 4)

def get_audio_file(ref_video_filename, offset, audio_filename):
    video = VideoFileClip(ref_video_filename)
    audio = video.audio 
    audioclip = audio.subclip(offset)
    audioclip.write_audiofile(audio_filename)

def main():
    # returns offset of two videos in seconds
    offset = calculate_offset("./edited_test.mp4", "./aya_hwasa.mp4")
    print(offset)

    # creates audio file given audio file name
    get_audio_file("./aya_hwasa.mp4", offset, "./aya_audio.mp3")

if __name__ == "__main__":
    main()