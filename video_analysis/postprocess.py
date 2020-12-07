# from moviepy.editor import *
# videoclip = VideoFileClip("filename.mp4")
# audioclip = AudioFileClip("audioname.mp3")

# new_audioclip = CompositeAudioClip([audioclip])
# videoclip.audio = new_audioclip
# videoclip.write_videofile("new_filename.mp4")

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, CompositeVideoClip, clips_array

def side_by_side_videos(test_video_file_path, ref_video_file_path, target_file_path):
    # aiming for 800 x 1200 pixels
    w0 = 800
    h0 = 1200
    desired_ratio = w0/h0
    
    clip1 = VideoFileClip(test_video_file_path) # add 30px contour
    clip2 = VideoFileClip(ref_video_file_path) # add 30px contour
    # final_clip = clips_array([[clip1], [clip2]]) # top bottom
    # final_clip = clips_array([[clip1, clip2]]) # side by side - this isn't working properly though not sure why
    # final_clip.resize(width=480).write_videofile(target_file_path)

    w1 = clip1.w
    h1 = clip1.h
    if w1/h1 > desired_ratio:
        # scale width
        clip1 = clip1.resize(w0/w1)
    else:
        # scale height
        clip1 = clip1.resize(h0/h1)

    w2 = clip2.w
    h2 = clip2.h
    if w2/h2 > desired_ratio:
        clip2 = clip2.resize(w0/w1)
    else:
        clip2 = clip2.resize(h0/h2)
    video = clips_array([[clip1,clip2]])
    video.write_videofile(target_file_path)
    # video = CompositeVideoClip([clip1.set_position((0,0)),
    #                        clip2.set_position((w0,0))])
    # video.resize(width=w0*2, height=h0).write_videofile(target_file_path)

    


def add_audio_to_video(video_file_path, audio_file_path, target_file_path):
    videoclip = VideoFileClip(video_file_path)
    audioclip = AudioFileClip(audio_file_path)
    audioclip = audioclip.subclip(0, videoclip.duration)

    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    videoclip.write_videofile(target_file_path)

def main():
    # audio = "../aya_audio.mp3"
    # video = "./media/aya_comparison_offset_video_short.mp4"
    # add_audio_to_video(video,audio,"./media/aya_combined_audio_video.mp4")

    side_by_side_videos("./media/test_output.mp4", "./media/reference_output.mp4", "./media/combined_openpose_output_side4.mp4")
    

if __name__ == "__main__":
    main()