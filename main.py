from utils import read_videos,save_video
from trackers import Tracker
import cv2 # type: ignore
from team_assigner import TeamAssigner

def main():
    #read video
    video_frames = read_videos('input_videos/B1606b0e6_1 (36).mp4')

    #Initialising Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracker(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Save the croped image of a player
    # for track_id, player in tracks['players'][0].items():
    #     bound_box = player['bbox']
    #     frame = video_frames[0]

    #     cropped_image = frame[int(bound_box[1]) : int(bound_box[3]), int(bound_box[0]) : int(bound_box[2])]

    #     #save the croped image
    #     cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
    #     break


    #Assign each player it's team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num , player_track in enumerate(tracks["players"]):
        for player_id , track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)

            tracks["players"][frame_num][player_id]['team'] = team
            tracks["players"][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw object tracker
    output_video_frames = tracker.draw_anotations(video_frames, tracks)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()