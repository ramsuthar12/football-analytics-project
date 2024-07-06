from utils import read_videos,save_video
from trackers import Tracker
import cv2 # type: ignore
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer

def main():
    #read video
    video_frames = read_videos('input_videos/08fd33_4.mp4')

    #Initialising Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracker(video_frames, read_from_stub=True, stub_path='stubs/track_stubs_2.pkl')

    #Get object positions
    tracker.add_position_to_tracks(tracks)

    #Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stubs=True, stubs_path='stubs/camera_movement_stub.pkl')

    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)

    #View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

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

    #Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()