import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def optimum_number_of_clusters(self, iterations=300):
    """
    :return: optimum number of clusters that would be used in the
    kMeans algorithm using the elbow method
    # default and suitable number of iterations for most dataset: 300
    """

    _wcss = []
    for i in range(1, 11):
        _kmeans = KMeans(
            n_clusters=i,
            init='k-means++',
            max_iter=iterations,
            n_init=10,
            random_state=0
        )
        _kmeans.fit(self.X)
        _wcss.append(_kmeans.inertia_)
    # Plotting the results onto a line graph, allowing us to observe
    # 'The elbow'
    plt.plot(range(1, 11), _wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.show()
