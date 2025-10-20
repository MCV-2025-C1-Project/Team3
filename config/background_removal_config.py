#  CONFIG BACKGROUND REMOVAL ALGORITHM X
#----------------------------------------------------------------------------------

# Step size (in pixels) for line scanning.
# Smaller values → denser scanning but slower processing.
SCAN_STEP = 2

# Number of pixels used to compute the baseline intensity
# at the beginning (or end) of each scan line.
# Higher values smooth initial noise but can blur edge detection.
EDGE_BOOTSTRAPS = [10]

# Size (in pixels) of the moving average window
# used to compute local mean intensity along each scan line.
RUN_WINDOWS = [2]

# Minimum required intensity difference (ΔI) between the baseline
# and a local window to consider it a valid edge transition.
# Larger values → only strong edges are detected.
DELTA_INTS = [50]
#DELTA_INTS = [35, 40, 45, 50, 55, 60, 65, 70]

# Number of consecutive windows that must confirm the same transition
# for it to be accepted as a valid edge (stability parameter).
CONSISTENCIES = [2]

# Maximum allowed deviation (in pixels) from the median position
# when filtering outlier edge detections along each side.
TOLS = [15]

# Variance threshold of the saturation channel.
# If var(S) < VAR_THRESHOLD, the algorithm switches to the luminance channel (Y)
# because the image is too uniform in color to rely on saturation.
VAR_THRESHOLD = 650

G_THRESHOLDS = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190]

SAVE_PLOTS = False








# CONFIG BACKGROUND REMOVAL ALGORITHM A