from utils import read_videos,save_video
from trackers import Tracker

def main():
    #read video
    video_frames = read_videos('input_videos/B1606b0e6_1 (36).mp4')

    #Initialising Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracker(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Draw object tracker
    output_video_frames = tracker.draw_anotations(video_frames, tracks)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()