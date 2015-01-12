import operator
import numpy as np

from dipy.segment.metric import Metric
from dipy.segment.metric import AveragePointwiseEuclideanMetric


class Identity:
    def __getitem__(self, idx):
        return idx


class Cluster(object):
    """ Provides functionalities to interact with a cluster.

    Useful container to retrieve index of elements grouped together. If
    a reference to the data is provided to `cluster_map`, elements will
    be returned instead of their index when possible.

    Parameters
    ----------
    cluster_map : `ClusterMap` object
        reference to the set of clusters this cluster is being part of
    id : int
        id of this cluster in its associated `cluster_map`

    Notes
    -----
    A cluster does not contain actual data but instead knows how to
    retrieves them using its `ClusterMap` object.
    """
    def __init__(self, id=0, indices=None, refdata=Identity()):
        self.id = id
        self.refdata = refdata
        self.indices = indices if indices is not None else []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, np.integer):
            return self.refdata[self.indices[idx]]
        elif type(idx) is slice:
            return [self.refdata[i] for i in self.indices[idx]]
        elif type(idx) is list:
            return [self[i] for i in idx]

        raise TypeError("Index must be a int or a slice! Not " + str(type(idx)))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __str__(self):
        return "[" + ", ".join(map(str, self.indices)) + "]"

    def __repr__(self):
        return "Cluster(" + str(self) + ")"

    def __eq__(self, other):
        return isinstance(other, Cluster) \
            and self.indices == other.indices

    def __ne__(self, other):
        return not self == other

    def __cmp__(self, other):
        raise TypeError("Cannot compare Cluster objects.")

    def assign(self, *indices):
        """ Assigns indices to this cluster.

        Parameters
        ----------
        *indices : list of indices
            indices to add to this cluster
        """
        self.indices += indices


class ClusterCentroid(Cluster):
    """ Provides functionalities to interact with a cluster.

    Useful container to retrieve index of elements grouped together and
    the cluster's centroid. If a reference to the data is provided to
    `cluster_map`, elements will be returned instead of their index when
    possible.

    Parameters
    ----------
    cluster_map : `ClusterMapCentroid` object
        reference to the set of clusters this cluster is being part of
    id : int
        id of this cluster in its associated `cluster_map`

    Notes
    -----
    A cluster does not contain actual data but instead knows how to
    retrieves them using its `ClusterMapCentroid` object.
    """
    def __init__(self, centroid, id=0, indices=None, refdata=Identity()):
        super(ClusterCentroid, self).__init__(id, indices, refdata)
        self.centroid = centroid.copy()
        self.new_centroid = centroid.copy()

    def __eq__(self, other):
        return isinstance(other, ClusterCentroid) \
            and np.all(self.centroid == other.centroid) \
            and super(ClusterCentroid, self).__eq__(other)

    def assign(self, id_datum, features):
        """ Assigns a data point to this cluster.

        Parameters
        ----------
        id_datum : int
            index of the data point to add to this cluster
        features : 2D array
            data point's features to modify this cluster's centroid
        """
        N = len(self)
        self.new_centroid = ((self.new_centroid * N) + features) / (N+1.)
        super(ClusterCentroid, self).assign(id_datum)

    def update(self):
        """ Update centroid of this cluster.

        Returns
        -------
        converged : bool
            tells if the centroid has moved
        """
        converged = np.equal(self.centroid, self.new_centroid)
        self.centroid = self.new_centroid.copy()
        return converged


class ClusterMap(object):
    """ Provides functionalities to interact with clustering outputs.

    Useful container to create, remove, retrieve and filter clusters.
    If `refdata` is given, elements will be returned instead of their
    index when using `Cluster` objects.

    Parameters
    ----------
    refdata : list
        actual elements that clustered indices refer to
    """
    def __init__(self, refdata=Identity()):
        self._clusters = []
        self.refdata = refdata

    @property
    def clusters(self):
        return self._clusters

    @property
    def refdata(self):
        return self._refdata

    @refdata.setter
    def refdata(self, value):
        self._refdata = value
        for cluster in self.clusters:
            cluster.refdata = self._refdata

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            return [self.clusters[i] for i, take_it in enumerate(idx) if take_it]
        elif type(idx) is list:
            return [self.clusters[i] for i in idx]

        return self.clusters[idx]

    def __iter__(self):
        return iter(self.clusters)

    def __str__(self):
        return "[" + ", ".join(map(str, self)) + "]"

    def __repr__(self):
        return "ClusterMap(" + str(self) + ")"

    def _richcmp(self, other, op):
        """ Compare
        Parameters
        ----------
        other:
        op: rich comparison operators
            lt, le, eq, ne, gt or ge (see module `operator`)
        """
        if isinstance(other, ClusterMap):
            if op is operator.eq:
                return isinstance(other, ClusterMap) \
                    and len(self) == len(other) \
                    and self.clusters == other.clusters
            elif op is operator.ne:
                return not self == other

            raise NotImplemented("Can only check if two ClusterMap instances are equal or not.")

        elif isinstance(other, int):
            return np.array([op(len(cluster), other) for cluster in self])

        raise NotImplemented("ClusterMap only supports comparison with a int or another instance of Clustermap.")

    def __eq__(self, other):
        return self._richcmp(other, operator.eq)

    def __ne__(self, other):
        return self._richcmp(other, operator.ne)

    def __lt__(self, other):
        return self._richcmp(other, operator.lt)

    def __le__(self, other):
        return self._richcmp(other, operator.le)

    def __gt__(self, other):
        return self._richcmp(other, operator.gt)

    def __ge__(self, other):
        return self._richcmp(other, operator.ge)

    def add_cluster(self, cluster):
        """ Adds a new cluster to this cluster map.

        Parameters
        ----------
        cluster : `Cluster` object
        """
        self.clusters.append(cluster)
        cluster.refdata = self.refdata

    def remove_cluster(self, cluster):
        """ Remove a cluster from this cluster map.

        Parameters
        ----------
        cluster : `Cluster` object
        """
        self.clusters.remove(cluster)

    def clear(self):
        """ Remove all clusters from this cluster map. """
        self.clusters.clear()


class ClusterMapCentroid(ClusterMap):
    """ Provides functionalities to interact with clustering outputs.

    Useful container to create, remove, retrieve and filter clusters.
    It allows also to retrieve easely the centroid of every clusters.
    If `refdata` is given, elements will be returned instead of their
    index when using `ClusterCentroid` objects.

    Parameters
    ----------
    refdata : list
        actual elements that clustered indices refer to
    """
    @property
    def centroids(self):
        return [cluster.centroid for cluster in self.clusters]


class Clustering:
    def cluster(self, data, ordering=None):
        """ Clusters `data`.

        Subclasses will perform their clustering algorithm here.

        Parameters
        ----------
        data : list of N-dimensional array
            each array represents a data point.
        ordering : iterable of indices
            change `data` ordering when applying the clustering algorithm

        Returns
        -------
        clusters : `ClusterMap` object
            result of the clustering
        """
        raise NotImplementedError("Subclass has to define this function!")


class QuickBundles(Clustering):
    r""" Clusters streamlines using QuickBundles algorithm.

    Given a list of streamlines, the QuickBundles algorithm sequentially
    assigns each streamline to its closest bundle in $\mathcal{O}(Nk)$ where
    $N$ is the number of streamlines and $k$ is the final number of bundles.
    If for a given streamline its closest bundle is farther than `threshold`,
    a new bundle is created and the streamline is assigned to it except if the
    number of bundles has already exceeded `max_nb_clusters`.

    Parameters
    ----------
    threshold : float
        The maximum distance from a bundle for a streamline to be still
        considered as part of it.
    metric : str or `Metric` object
        The distance metric to use. The metric can be 'mdf'.
    max_nb_clusters : int
        Limits the creation of bundles.

    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """
    def __init__(self, threshold, metric="mdf", max_nb_clusters=np.iinfo('i4').max):
        self.threshold = threshold
        self.max_nb_clusters = max_nb_clusters

        if isinstance(metric, Metric):
            self.metric = metric
        elif metric.lower() == "mdf":
            self.metric = AveragePointwiseEuclideanMetric()

    def cluster(self, streamlines, ordering=None, refdata=None):
        """ Clusters `streamlines` into bundles.

        Performs quickbundles algorithm using predefined metric and threshold.

        Parameters
        ----------
        streamlines : list (or generator) of 2D array
            each 2D array represents a sequence of 3D points (points, 3).
        ordering : iterable of indices
            change `streamlines` ordering when applying QuickBundles

        Returns
        -------
        clusters : `ClusterMapCentroid` object
            result of the clustering
        """
        from dipy.segment.clustering_algorithms import quickbundles
        cluster_map = quickbundles(streamlines, self.metric,
                                   threshold=self.threshold,
                                   max_nb_clusters=self.max_nb_clusters,
                                   ordering=ordering)
        if refdata is not None:
            cluster_map.refdata = streamlines

        return cluster_map