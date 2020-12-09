from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sys
sys.path.append("../")
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
        amount = result[i]["trim"]

    return round(amount, 4)

def get_audio_file(test_video_filename, offset, target_audio_filepath):
    video = VideoFileClip(test_video_filename)
    audio = video.audio 
    audioclip = audio.subclip(offset, video.duration)
    audioclip.write_audiofile(target_audio_filepath)

def get_duration(test_video_filename, ref_video_filename, offset):
    test = VideoFileClip(test_video_filename)
    ref = VideoFileClip(ref_video_filename)
    return min(test.duration, ref.duration - offset)

def trim_start_of_video(video_filename, offset, duration, target_name):
    ffmpeg_extract_subclip(video_filename, offset, offset+duration, targetname=target_name)

def main():
    # returns offset of two videos in seconds
    offset = calculate_offset("./edited_test.mp4", "./aya_hwasa.mp4")
    print(offset)

    # creates audio file given audio file name
    get_audio_file("./aya_hwasa.mp4", offset, "./aya_audio.mp3")

    # get_duration("../data/reference.mp4")

if __name__ == "__main__":
    main()