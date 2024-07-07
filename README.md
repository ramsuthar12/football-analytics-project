# Football Analytics Project

## Introduction
The objective of this project is to develop an advanced system for detecting and tracking players, referees, and footballs in video footage using YOLO, a state-of-the-art AI object detection model. Beyond detection, the project aims to enhance the model's performance through training and to assign players to teams based on t-shirt colors using K-means for pixel segmentation and clustering. This allows for the analysis of ball possession percentages for each team during a match.

Additionally, we employ optical flow to measure camera movement between frames, ensuring precise tracking of player movement. Perspective transformation is utilized to represent the depth and perspective of the scene, enabling us to measure player movement in meters rather than pixels. This facilitates accurate calculation of a player's speed and the distance they cover. This comprehensive project addresses real-world problems and incorporates various concepts, making it suitable for both beginners and experienced machine learning engineers.

## Modules Used
The following modules and techniques are utilized in this project:

- **YOLO**: AI object detection model for detecting players, referees, and footballs.
- **K-means**: Pixel segmentation and clustering to identify t-shirt colors and assign players to teams.
- **Optical Flow**: Technique to measure camera movement and ensure accurate player tracking.
- **Perspective Transformation**: Method to represent scene depth and perspective, allowing measurement of player movement in meters.
- **Speed and Distance Calculation**: Techniques to calculate each player's speed and the distance covered during a match.

## Requirements
To run this project, ensure the following dependencies are installed:

- Python 3.x
- Ultralytics
- Supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Input Video
The input video for this project can be downloaded from the following link: [Input Video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view)
