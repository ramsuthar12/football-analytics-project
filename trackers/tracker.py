from ultralytics import YOLO  # type: ignore
import supervision as sv # type: ignore
import pickle
import numpy as np
import pandas as pd
import os
import cv2 # type: ignore
import sys 
sys.path.append('../')
from utils import get_bound_box_centre, bound_box_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1,{}).get("bbox",[]) for x in ball_positions]
        # ball_positions = [x[key].get("bbox", []) if x else [] for x in ball_positions for key in (x if x else [None])]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        # return df_ball_positions
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i+batch_size], conf = 0.1)
            detections += detections_batch

        return detections

    def get_object_tracker(self, frames, read_from_stub = False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)

            print("tracks are retured from stubs")
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            #convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert goalkeeper to player object
            for obj_ind , cls_id in enumerate(detection_supervision.class_id): 
                if cls_names[cls_id] == "goalkeeper":
                    detection_supervision.class_id[obj_ind] = cls_names_inv["player"]

            #Track the objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bound_box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bound_box}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bound_box}

            for frame_detection in detection_supervision:
                bound_box = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bound_box}
        
        if stub_path is not None:
            with open(stub_path,'wb') as f :
                pickle.dump(tracks,f)

        print("new tracks created and returned")
        return tracks

    def draw_ellipse(self, frame, bound_box, color, track_id = None):
        y2 = int(bound_box[3])
        x_centre, _ = get_bound_box_centre(bound_box)
        width = bound_box_width(bound_box)

        cv2.ellipse(
            img=frame,
            center=(x_centre, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_centre - rectangle_width//2
        x2_rect = x_centre + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                            (int(x1_rect),int(y1_rect)),
                            (int(x2_rect),int(y2_rect)),
                            color,
                            cv2.FILLED
            )

            x1_text = x1_rect + 10
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2 
            )

        return frame

    def draw_triangle(self, frame, bound_box, color):
        y = int(bound_box[1])
        x,_ = get_bound_box_centre(bound_box)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_team_ball_position(self, frame , frame_num , team_ball_position):
        #Drawing a semi-transparent rectange to show the ball position of each team 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (0,0,0), cv2.FILLED)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        team_ball_position_till_the_frame = team_ball_position[:frame_num + 1]

        #Calculating ball percentage
        team_1_num_frames = team_ball_position_till_the_frame[team_ball_position_till_the_frame == 1].shape[0]
        team_2_num_frames = team_ball_position_till_the_frame[team_ball_position_till_the_frame == 2].shape[0]

        team_1 = team_1_num_frames/(team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Position : {team_1 * 100:.2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Team 2 Ball Position : {team_2 * 100:.2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return frame

    def draw_anotations(self, video_frames, tracks, team_ball_position):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            #Drawing the ellipse for the players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (248, 255, 36))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                #Drawing marker for the player with the ball 
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            #Drawing the ellipse for the referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (36, 255, 248))

            #Drawing the triangle for the ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))
            

            #Drawing Ball Position for each team
            frame = self.draw_team_ball_position(frame, frame_num, team_ball_position)

            output_video_frames.append(frame)

        return output_video_frames

