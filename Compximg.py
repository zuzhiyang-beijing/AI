from os import listdir
import cv2
from PIL import Image
from moviepy.editor import *
if __name__ == '__main__':
    pic_dir = r'/shengri/1.jpg'
    mp3_path = r'/shengri/1.mp3'
    final_video_path = r'/shengri/1.mp4'
    img = Image.open(pic_dir)
    mp3_clip = AudioFileClip(mp3_path)
    img_clips = []
    img_clips.append(ImageClip(pic_dir,duration= mp3_clip.duration))
    result_video = concatenate_videoclips(img_clips)
    result_video = result_video.set_audio(mp3_clip)
    result_video.write_videofile(final_video_path,fps=24,audio_codec='aac')
    #result_video.write_videofile(final_video_path,fps=25,audio_codec="libmp3lame")
    result_video.close()
