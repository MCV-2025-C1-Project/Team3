DENOISING_GRID = {
    "Gaussian": {
        "ksize": [(3, 3), (5, 5), (7, 7)],
        "sigma": [0.5, 1.0, 1.5, 2.0]
    },

    "Median": {
        "ksize": [3, 5, 7, 9]
    },

    "Bilateral": {
        "d": [5, 9, 15],
        "sigmaColor": [25, 50, 75, 100],
        "sigmaSpace": [25, 50, 75, 100]
    },

    "non_local_means": {
        "h": [0.6, 0.8, 1.0, 1.2],
        "patch_size": [3, 5, 7],
        "patch_distance": [5, 9, 11],
        "fast_mode": [True]
    },

    "bm3d": {
        "sigma_psd": [15/255.0, 25/255.0, 35/255.0, 50/255.0]
    }
}

NOISE_REMOVAL = "Gaussian"
