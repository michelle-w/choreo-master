This project uses pretrained Stacked Hourglass and OpenPose pose detection models to analyze and provide feedback on choreography videos. A user can compare their video to the reference video and receive a feedback report.

### Motivation and inspiration
For this project, we are specifically interested in exploring human pose estimation with single-person videos.This project aims to detect poses in a given video and a reference video, and then with the detected keypoints, quantify differences between the two videos for the purposes of scoring how well the given video follows the poses of the original reference video. Specifically, we create a video analysis pipeline that takes two single-person videos as input, compares the poses of the individuals in the videos based on two metrics, and outputs specific segments and frames with the largest differences between the two videos. 

## Sample Input/Output Data
See this [Google Drive folder](https://drive.google.com/drive/folders/1KAC-0UknWPeNyG1fJG5VcfPp1luRhbC6?usp=sharing) for an example project with inputs and outputs, as well as other reference videos created throughout the process of working on this project.

## How to use this repository

### Assumptions
We assume that the reference video is longer than the test video--the offset between videos and generated audio are based on the reference video clip. 
Pose estimation models work best when the input is of a single individual who is centered in the video and the video is cropped to the individual. 
Feedback is provided in the form of the problem sections that score below the 25th percentile of all scores, and the top n half second sections that had the lowest scores, where the scores are given by a specified metric.

### Analyze using Stacked Hourglass model
The main script in `video_analyzer.py` shows an example run using Stacked Hourglass for inference. Creating a Video_Analyzer object requires a name for the choreo you are analyzing, two video filenames (one for reference, one for test), as well as a base path for saving your output(s) to, and desired frame rate. 

### Analyze using OpenPose
Call the `initial_processing_for_openpose` function in the `Video_Analyzer` class found in the `video_analyzer.py` folder to generate the resulting audio and trimmed reference video based on the offset between the test and reference videos.
For the rest of the pipeline, we recommend downloading the `Pose_Detection_with_OpenPose_and_Video_Feedback.ipynb` notebook and running it on Google Colab. Upload the test video, reference video, and audio file to the Colab session, and make sure that Runtime is set to GPU. Use the Colab notebook to generate predictions and output videos. 
Here is an overview:
<ul>
<li> Install OpenPose (~14 minutes) </li>
<li>Run OpenPose on videos (~2 minutes for a 20 second video) </li>
<li>Extraction of frame data  </li>
<li>Score analysis </li>
<li>Post processing (~3 minutes to generate merged video and add audio) </li>
</ul>


## References
See the [OpenPose]() and [StackedHourglass](https://arxiv.org/pdf/1603.06937.pdf) papers and implementation Github code below.

Specific libraries used: [Moviepy](https://zulko.github.io/moviepy/index.html) for editing audio and video clips, [align videos by sound](https://github.com/jeorgen/align-videos-by-sound) for aligning the test and reference videos.

<!-- ### GitHub repositories consulted
Hourglass Model
<ul>
<li> Original paper code for Stacked Hourglass (2016): <a href="https://github.com/princeton-vl/pose-hg-train)">GitHub repository</a>
<li> PyTorch implementation of Stacked Hourglass (2018): [pytorch_stacked_hourglass](https://github.com/princeton-vl/pytorch_stacked_hourglass) </li>
<li>Pretrained Stacked Hourglass Model: [PyTorch toolkit for 2D estimation](https://github.com/anibali/pytorch-stacked-hourglass)</li>
<li>Helper functions for evaluation and plotting keypoints for Hourglass: [TensorFlow Hourglass]https://github.com/ethanyanjiali/deep-vision/tree/master/Hourglass/tensorflow</li>
</ul>
OpenPose
<ul>
<li> GitHub repository: [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) </li>
<li> Pose Detection with OpenPose demo and other Colab notebooks: [pose detection notebooks](https://github.com/tugstugi/dl-colab-notebooks#pose-detection)
</ul> -->






