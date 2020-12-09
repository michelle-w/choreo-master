import sys
sys.path.append("../")

import preprocess as preprocess
import input_processing_MPII as process_MPII
import input_processing_body25 as process_body25
import postprocess as postprocess

import numpy as np
import os

from stacked_hourglass.infer import get_keypoints
from score_analysis import find_problem_times, find_top_n_problems, generate_feedback_report

class Video_Analyzer:
    '''
    test_path = video file name
    '''
    def __init__(self, name, test_path, ref_path, base_path, fps):
        self.name = name
        self.test_path = test_path
        self.ref_path = ref_path
        self.base_path = base_path
        self.fps = fps
        self.path = os.path.join(base_path, name)

        # create folder for this input
        try:  
            # creating a folder
            if not os.path.exists(self.path):
                os.makedirs(self.path)

        # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data') 
        
    def read_inputs_for_hourglass(self):
        self.offset = preprocess.calculate_offset(os.path.join(self.path, self.test_path), os.path.join(self.path, self.ref_path))
        preprocess.get_audio_file(os.path.join(self.path, self.ref_path), self.offset, os.path.join(self.path, "{}_audio.mp3".format(self.name)))

        self.duration = preprocess.get_duration(os.path.join(self.path, self.test_path), os.path.join(self.path, self.test_path), self.offset)
        
        # extract frames for StackedHourglass
        path = os.path.join(self.base_path, "{}".format(self.name))
        process_MPII.extract_frames(self.test_path, start_time = 0, duration = self.duration *1000, desired_fps = 24, target_folder_name = "test_frames", base_path = path, max_frames = 10000)
        process_MPII.extract_frames(self.ref_path, start_time = self.offset, duration = self.duration*1000, desired_fps = 24, target_folder_name = "ref_frames", base_path = path, max_frames = 10000)
    
    def analyze_frames_for_hourglass(self):
        path = os.path.join(self.base_path, "{}".format(self.name))
        rel_no_roms_accs, rel_roms_accs, abs_accs = process_MPII.process_frames(
            "test_frames", "ref_frames", 
            num_frames = int(self.duration * self.fps), 
            target_folder_name = "sh_comparison_frames", base_path = path, 
            offset = self.offset, 
            save = True)
        
        target_file_name = os.path.join(path,self.name+"_rel_no_roms_accs_sh.npy")
        np.save(target_file_name, rel_no_roms_accs)
        target_file_name = os.path.join(path,self.name+"_rel_roms_accs_sh.npy")
        np.save(target_file_name, rel_roms_accs)
        target_file_name = os.path.join(path,self.name+"_abs_accs_sh.npy")
        np.save(target_file_name, abs_accs)
    
    def generate_output_for_hourglass(self):
        postprocess.merge_frames_to_video_sh(
            frame_folder_path=os.path.join(self.base_path, self.name, "sh_comparison_frames"),
            target_file_path=os.path.join(self.base_path, self.name, "sh_comparison.mp4"),
            fps = 24,
            num_frames = self.duration * self.fps
        )

        postprocess.add_audio_to_video(
            video_file_path=os.path.join(self.base_path, self.name, "sh_comparison.mp4"),
            audio_file_path=os.path.join(self.base_path, self.name, "{}_audio.mp3".format(self.name)),
            target_file_path=os.path.join(self.base_path, self.name, "sh_comparison_with_audio.mp4")
        )


    def initial_processing_for_openpose(self):
        self.offset = preprocess.calculate_offset(os.path.join(self.path, self.test_path), os.path.join(self.path, self.ref_path))
        preprocess.get_audio_file(os.path.join(self.path, self.ref_path), self.offset, os.path.join(self.path, "{}_audio.mp3".format(self.name)))

        self.duration = preprocess.get_duration(os.path.join(self.path, self.test_path), os.path.join(self.path, self.ref_path), self.offset)
        
        preprocess.trim_start_of_video(os.path.join(self.path, self.ref_path), self.offset, self.duration, os.path.join(self.path, "trimmed_{}".format(self.ref_path)))


    def keypoint_processing_for_openpose(self, folder_name1, folder_name2, num_frames):
        rel_no_roms_accs, rel_roms_accs, abs_accs = process_body25.process_openpose_frames(self.name, folder_name1, folder_name2, num_frames, self.base_path)
        target_file_name = os.path.join(self.path,self.name+"_rel_no_roms_accs_op.npy")
        np.save(target_file_name, rel_no_roms_accs)
        target_file_name = os.path.join(self.path,self.name+"_rel_roms_accs_op.npy")
        np.save(target_file_name, rel_roms_accs)
        target_file_name = os.path.join(self.path,self.name+"_abs_accs_op.npy")
        np.save(target_file_name, abs_accs)

    def generate_output_for_openpose(self):
        postprocess.side_by_side_videos(
            test_video_file_path="{}_test_output.mp4".format(self.name),
            ref_video_file_path="{}_reference_output.mp4".format(self.name),
            target_file_path=os.path.join(self.base_path, self.name, "op_comparison.mp4")
        )

        postprocess.add_audio_to_video(
            video_file_path=os.path.join(self.base_path, self.name, "op_comparison.mp4"),
            audio_file_path=os.path.join(self.base_path, self.name, "{}_audio.mp3".format(self.name)),
            target_file_path=os.path.join(self.base_path, self.name, "op_comparison_with_audio.mp4")
        )
        

    def get_problem_times(self, accs_file_name, fps, percentile=25):
        accs = np.load(accs_file_name)
        return find_problem_times(accs, fps, percentile)
        
    def get_top_n_problems(self, accs_file_name, fps, n):
        accs = np.load(accs_file_name)
        return find_top_n_problems(accs, fps, n)

    def get_feedback_report(self, accs_file_name, fps):
        accs = np.load(os.path.join(self.base_path, self.name, accs_file_name))
        report = generate_feedback_report(self.name, accs, fps)
        text_file = open(os.path.join(self.base_path, self.name, "{}_report.txt".format(self.name)), "w")
        n = text_file.write(report)
        text_file.close()
        print(accs)


def main():
    name = "blackpink"
    # name = "seven_rings"
    base_path = "../data"
    # test_path = "seven_rings_test.mp4"
    # ref_path = "seven_rings_ref.mp4"
    test_path = "test.mp4"
    ref_path = "reference.mp4"
    fps = 24
    test_analyzer = Video_Analyzer(name, test_path, ref_path, base_path, fps)
    test_analyzer.read_inputs_for_hourglass()
    test_analyzer.analyze_frames_for_hourglass()
    test_analyzer.generate_output_for_hourglass()
    test_analyzer.get_feedback_report(name+"_rel_roms_accs_sh.npy", 24)

    test_analyzer.initial_processing_for_openpose()



if __name__ == "__main__":
    main()