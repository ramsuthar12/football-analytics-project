from utils import read_videos,save_video
from trackers import Tracker
import cv2 # type: ignore
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    #read video
    video_frames = read_videos('input_videos/08fd33_4.mp4')

    #Initialising Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracker(video_frames, read_from_stub=True, stub_path='stubs/track_stubs_2.pkl')

    # Save the croped image of a player
    # for track_id, player in tracks['players'][0].items():
    #     bound_box = player['bbox']
    #     frame = video_frames[0]

    #     cropped_image = frame[int(bound_box[1]) : int(bound_box[3]), int(bound_box[0]) : int(bound_box[2])]

    #     #save the croped image
    #     cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
    #     break

    #Interpolate ball positions 
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    #Assign each player it's team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num , player_track in enumerate(tracks["players"]):
        for player_id , track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)

            tracks["players"][frame_num][player_id]['team'] = team
            tracks["players"][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    #Assigning the player with the ball
    player_assigner = PlayerBallAssigner()
    team_ball_position = []

    for frame_num , player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_position.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_position.append(team_ball_position[-1])

    team_ball_position = np.array(team_ball_position)

    # Draw object tracker
    output_video_frames = tracker.draw_anotations(video_frames, tracks, team_ball_position)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()