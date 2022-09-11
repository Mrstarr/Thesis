from this import d
import numpy as np
from sklearn import cluster


class PathCluster():

    def __init__(self,k, max_iter) -> None:
        self.k = k
        self.max_iter = max_iter


    def cluster(self, paths):
        clustered_paths = []

        for path in paths:
            centroid = self.initialize(path)
            iter = 0
            converged = False
            
            while not converged and iter < self.max_iter:

                clusters = self.nearest_cluster(path, centroid)
                new_centroid = self.update_centroid(path, clusters)

                dist = np.sum(np.linalg.norm(centroid-new_centroid,axis=2))
                if dist < 1e-5:
                    converged = True
            
                centroid = new_centroid
                iter += 1

            clt_path = self.centroid_path(path, centroid)
            clustered_paths.append(clt_path)

        return clustered_paths
        # steer given centroids

        # return steered clustered paths
    

    def nearest_cluster(self, path, centroid):
        clusters = []
        for j in range(self.k):
            clusters.append([])
        for p in path:
            p_tr = np.array([v.pose[0:2] for v in p])
            cluster_idx = np.argmin(np.sum(np.linalg.norm(centroid-p_tr,axis=2),axis=1))
            clusters[cluster_idx].append(p_tr)
        return clusters


    def update_centroid(self, path, clusters):
        centroid = []
        for cluster in clusters:
            # prevent cluster NAN condition
            if cluster:
                centroid.append(np.mean(np.array(cluster), axis = 0))
            else:
                centroid.append(np.array([v.pose[0:2] for v in path[np.random.randint(0, len(path))]]))

        return np.array(centroid)


    def initialize(self, paths):
        path_len = len(paths)
        rand_idx = np.random.randint(0, path_len)
        centroid = [[v.pose[0:2] for v in paths[rand_idx]]]
        #centroid = [[v[0:2] for v in paths[rand_idx]]]
        
        i = 1
        while i < self.k:
            max_dis = 0
            for path in paths:
                p = [v.pose[0:2] for v in path]
                #p = [v[0:2] for v in path]

                dist = min([self.euclidean(c, p) for c in centroid])
                if dist > max_dis:
                    max_dis = dist
                    new_centroid = p
            centroid.append(new_centroid)
            i+= 1
        return np.array(centroid)
    

    def centroid_path(self, path, centroid):
        clt_path = []
        for c in centroid:
            min_dis = 100
            for p in path:
                p_tr = [v.pose[0:2] for v in p]
                dist = self.euclidean(c,p_tr)
                if dist < min_dis:
                    min_dis = dist
                    min_clt = p
            clt_path.append(min_clt)
        return clt_path

    
    def euclidean(self, path1, path2):
        return np.sum(np.linalg.norm(np.array(path2)-np.array(path1), axis=1))


# paths = [[(1,1,1),(1,2,1),(1,3,1)],
#         [(1,1,1),(2,1,1),(3,1,1)],
#         [(1,1,1),(1.5,1.5,1),(2,2.5,1)],
#         [(1,1,1),(1.5,1.5,1),(2.2,2.2,1)],
#         [(1,1,1),(2,1,1),(2.5,1.5,1)],
#         [(1,1,1),(1.5,1.5,1),(2.5,2,1)]
#         ]
    
# pc = PathCluster()
# pc.cluster(paths)