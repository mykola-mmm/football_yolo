from utils import *

def main():
    video_frames, fps  = read_video('input_vids/08fd33_4.mp4')
    save_video(video_frames, 'output_vids/08fd33_4.avi', fps)

if __name__ == '__main__':
    main()