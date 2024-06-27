from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}



    def get_clustering_model(self, image):
        image_2d = image.reshape(-1,3)

        k_means = KMeans(n_clusters=2, init="k-means++", n_init=10)
        k_means.fit(image_2d)

        return k_means

    def get_player_color(self, frame, bound_box):
        image = frame[int(bound_box[1]):int(bound_box[3]) , int(bound_box[0]):int(bound_box[2])]
        
        top_half_image = image[0: int(image.shape[0]/2), :]

        #Getting the kmeans model 
        k_means = self.get_clustering_model(top_half_image)

        #getting the cluster_labels and reshaping to the original image shape
        labels = k_means.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        #getting the player's cluster
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = k_means.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self, frame, player_detections):

        player_colors = []
        for _, player_detection in player_detections.items():
            bound_box = player_detection["bbox"]
            player_color = self.get_player_color(frame, bound_box)
            player_colors.append(player_color)

        k_means = KMeans(n_clusters=2, init="k-means++", n_init=10)
        k_means.fit(player_colors)

        self.kmeans = k_means

        self.team_colors[1] = k_means.cluster_centers_[0]
        self.team_colors[2] = k_means.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]

        team_id += 1

        if player_id == 109:
            team_id = 2

        self.player_team_dict[player_id] = team_id

        return team_id

