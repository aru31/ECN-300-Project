import math
import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


class LinguisticSummary(object):
    """
    Generic class that applies K means algorithm and
    outputs meaningful sentences
    """

    def __init__(self, X=None, Y=None, num_clusters=None, num_features=None):
        """
        Constructor class
        """

        self.X = X
        self.Y = Y
        self.kmeans = None
        self.ykmeans = None
        self.num_clusters = num_clusters
        self.target_names = None
        self.num_features = num_features
        self.cluster_formation = None
        self.cluster_centers = None
        self.euclid_dist_from_center = None

        self.kmeans_algorithm()

    def kmeans_algorithm(self, iterations=300):
        """
        :param iterations: default 300, for best results use same as in for the
        elbow method
        :return: K means instance
        """

        self.kmeans = KMeans(
            n_clusters=self.num_clusters,
            init='k-means++',
            max_iter=iterations,
            n_init=10,
            random_state=0
        )
        self.ykmeans = self.kmeans.fit_predict(self.X)
        self.cluster_centers = self.kmeans.cluster_centers_

        self.get_cluster_points()
        # print(self.cluster_centers)

    def get_cluster_points(self):
        """
        Separate the cluster points according to their centers
        """

        diff_targets = set()
        for _targets in self.ykmeans:
            diff_targets.add(_targets)
        self.cluster_formation = dict()

        for target in diff_targets:
            data = list()
            self.cluster_formation[target] = list()
            for j in range(0, len(self.X[self.ykmeans == target, ])):
                data_point = list()
                for i in range(0, self.num_features):
                    data_point.append(self.X[self.ykmeans == target, i][j])
                data.append(data_point)

            self.cluster_formation[target] = data

        # from pprint import pprint
        # pprint(self.cluster_formation)
        self.get_euclid_dist()

    def euclidean_distance(self, data_point, center):
        """
        :param data_point: data_point
        :param center: cluster center
        :return euclidean distance
        """
        distance = 0
        for i in range(0, len(data_point)):
            distance = distance + (data_point[i]-center[i])**2
        return math.sqrt(distance)

    def get_center_by_average(self, data):
        """
        :param data: data
        :return center by average of all the points
        Generic average data point of all clusters
        This center will be almost same as the Kmeans calculated center
        """

        average_distance = []
        total = len(data)
        data_length = data[0]

        for i in range(0, len(data_length)):
            dist = 0
            for data_point in data:
                dist = dist + data_point[i]
            average_distance.append(dist / total)
        return average_distance

    def closest_center(self, average_center):
        """
        :param average_center: average_center
        :return the closest center by checking
        the minimum distance
        """

        index_array = []
        for i in range(0, len(self.cluster_centers)):
            index_array.append([
                i,
                self.euclidean_distance(
                    average_center,
                    self.cluster_centers[i]
                )
            ])
        min1 = 1000000000
        for i in range(0, len(index_array)):
            min1 = min(min1, index_array[i][1])

        index = -1
        for point in index_array:
            if point[1] == min1:
                index = point[0]
                break

        return self.cluster_centers[index], index

    def all_distance_from_clusters(self, center, data):
        """
        :param center: center
        :param data: data
        get euclidean distance from the center
        """

        dist_arr = []
        for point in data:
            dist_arr.append(self.euclidean_distance(point, center))
        return dist_arr

    def get_euclid_dist(self):
        """
        :return: euclid distance from center
        """

        self.euclid_dist_from_center = dict()

        for key in self.cluster_formation.keys():
            average_center = self.get_center_by_average(
                self.cluster_formation[key]
            )
            get_close_center, index = self.closest_center(average_center)

            self.euclid_dist_from_center[
                index
            ] = self.all_distance_from_clusters(
                        get_close_center, self.cluster_formation[key])

        # from pprint import pprint
        # pprint(self.euclid_dist_from_center)

    def silhouette_score(self):
        """
        :return: silhouette score
        """

        sc = silhouette_score(self.X, self.kmeans.labels_)
        return sc

    def num_of_points_within(self, max_dis, data):
        """
        :max_dis: max_distant point of the cluster
        :data: data
        number of points within the 90 % of the cluster
        in all the clusters
        """
        tot = 0
        for dist in data:
            if dist < (0.9 * max_dis):
                tot = tot + 1
        return tot

    def linguistic_summary(self):
        """
        :return: generic linguistic summary of the dataset
        """

        print(f"There are {self.num_clusters} clusters")
        for cluster in range(0, self.num_clusters):
            print(
                f"The cluster {cluster+1} centre are found to "
                f"be {self.cluster_centers[cluster]}"
            )

        for cluster in self.euclid_dist_from_center.keys():
            max_dis = np.max(self.euclid_dist_from_center[cluster])
            min_dis = np.min(self.euclid_dist_from_center[cluster])
            var = np.var(self.euclid_dist_from_center[cluster])
            print(f"For cluster {cluster+1}, maximum data point distance is {max_dis}")
            print(f"For cluster {cluster+1}, maximum data point distance is {min_dis}")
            print(f"For cluster {cluster+1}, variance is {var}")

        print(f"The labels for each of them are {self.kmeans.labels_}")
        print(f"The Silhouette score is {self.silhouette_score()}")
        print("Considering points outside 90% radius of cluster as outliers: ")

        for cluster in self.euclid_dist_from_center.keys():
            max_dis = np.max(self.euclid_dist_from_center[cluster])
            outliers = self.num_of_points_within(
                max_dis, self.euclid_dist_from_center[cluster]
            )
            total = len(self.euclid_dist_from_center[cluster])
            print(f"For cluster {cluster+1}, outliers are {total-outliers} out of {total}")
