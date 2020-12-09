from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, CompositeVideoClip, clips_array
import cv2

def side_by_side_videos(test_video_file_path, ref_video_file_path, target_file_path):
    # aiming for 800 x 1200 pixels
    w0 = 800
    h0 = 1200
    desired_ratio = w0/h0
    
    clip1 = VideoFileClip(test_video_file_path)
    clip2 = VideoFileClip(ref_video_file_path)

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

    
def merge_frames_to_video_sh(frame_folder_path, target_file_path, fps, num_frames):
    i = 0
    img = cv2.imread(frame_folder_path + "/frame{}.png".format(i))
    height, width, layers = img.shape
    size = (width,height)
    out = cv2.VideoWriter(target_file_path,cv2.VideoWriter_fourcc(*'FMP4'), fps, size)

    for i in range(num_frames):
        img = cv2.imread(frame_folder_path + "/frame{}.png".format(i))
        out.write(img)
    
    out.release() 

def add_audio_to_video(video_file_path, audio_file_path, target_file_path):
    videoclip = VideoFileClip(video_file_path)
    audioclip = AudioFileClip(audio_file_path)
    audioclip = audioclip.subclip(0, videoclip.duration)

    new_audioclip = CompositeAudioClip([audioclip])
    # videoclip.audio = new_audioclip
    videoclip2 = videoclip.set_audio(new_audioclip)
    videoclip2.write_videofile(target_file_path, codec="libx264", audio_codec="aac")

def main():
    # audio = "../aya_audio.mp3"
    # video = "./media/aya_comparison_offset_video_short.mp4"
    # add_audio_to_video(video,audio,"./media/aya_combined_audio_video.mp4")

    side_by_side_videos("./media/test_output.mp4", "./media/reference_output.mp4", "./media/combined_openpose_output_side4.mp4")
    

if __name__ == "__main__":
    main()