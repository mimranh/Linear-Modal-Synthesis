import numpy as np


class CrossResolutionValidator:

    def __init__(self,
                 freq_tol=20,
                 damping_tol=200,
                 min_support=2):

        self.freq_tol = freq_tol
        self.damping_tol = damping_tol
        self.min_support = min_support

    def cluster_modes(self, features_per_resolution):

        all_modes = []

        for res_modes in features_per_resolution:
            for m in res_modes:
                all_modes.append(m)

        clusters = []

        for mode in all_modes:
            placed = False

            for cluster in clusters:
                if abs(mode["f"] - cluster[0]["f"]) < self.freq_tol:
                    cluster.append(mode)
                    placed = True
                    break

            if not placed:
                clusters.append([mode])

        return clusters

    def validate_clusters(self, clusters):

        validated_modes = []

        for cluster in clusters:

            if len(cluster) < self.min_support:
                continue

            freqs = np.array([m["f"] for m in cluster])
            dampings = np.array([m["d"] for m in cluster])
            amps = np.array([m["A"] for m in cluster])

            f_mean = np.mean(freqs)
            d_mean = np.mean(dampings)
            A_mean = np.mean(amps)

            f_std = np.std(freqs)
            d_std = np.std(dampings)

            if f_std < self.freq_tol and d_std < self.damping_tol:
                validated_modes.append({
                    "f": f_mean,
                    "d": d_mean,
                    "A": A_mean,
                    "f_std": f_std,
                    "d_std": d_std,
                    "support": len(cluster)
                })

        return validated_modes