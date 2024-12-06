import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.linalg import svd

import mne


def setup_3d_axes():
    ax = plt.axes(projection="3d")
    ax.view_init(azim=-105, elev=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_zlim(0, 5)
    return ax

#A projection is an operation that converts one set of points into another set of points and repeating the projection operation on the resulting points has no effect. 
#ex1:
ax = setup_3d_axes()

# plot the vector (3, 2, 5)
origin = np.zeros((3, 1))
point = np.array([[3, 2, 5]]).T
vector = np.hstack([origin, point])
ax.plot(*vector, color="k")
ax.plot(*point, color="k", marker="o")

# project the vector onto the x,y plane and plot it
xy_projection_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
projected_point = xy_projection_matrix @ point
projected_vector = xy_projection_matrix @ vector
ax.plot(*projected_vector, color="C0")
ax.plot(*projected_point, color="C0", marker="o")

# add dashed arrow showing projection
arrow_coords = np.concatenate([point, projected_point - point]).flatten()
ax.quiver3D(
    *arrow_coords,
    length=0.96,
    arrow_length_ratio=0.1,
    color="C1",
    linewidth=1,
    linestyle="dashed",
)

#projections in average direction to remove effect of vector:
trigger_effect = np.array([[3, -1, 1]]).T
# compute the plane orthogonal to trigger_effect
x, y = np.meshgrid(np.linspace(-1, 5, 61), np.linspace(-1, 5, 61))
A, B, C = trigger_effect
z = (-A * x - B * y) / C
# cut off the plane below z=0 (just to make the plot nicer)
mask = np.where(z >= 0)
x = x[mask]
y = y[mask]
z = z[mask]

# compute the projection matrix
U, S, V = svd(trigger_effect, full_matrices=False)
trigger_projection_matrix = np.eye(3) - U @ U.T

# project the vector onto the orthogonal plane
projected_point = trigger_projection_matrix @ point
projected_vector = trigger_projection_matrix @ vector

# plot the trigger effect and its orthogonal plane
ax = setup_3d_axes()
ax.plot_trisurf(x, y, z, color="C2", shade=False, alpha=0.25)
ax.quiver3D(
    *np.concatenate([origin, trigger_effect]).flatten(),
    arrow_length_ratio=0.1,
    color="C2",
    alpha=0.5,
)

# plot the original vector
ax.plot(*vector, color="k")
ax.plot(*point, color="k", marker="o")
offset = np.full((3, 1), 0.1)
ax.text(*(point + offset).flat, "({}, {}, {})".format(*point.flat), color="k")

# plot the projected vector
ax.plot(*projected_vector, color="C0")
ax.plot(*projected_point, color="C0", marker="o")
offset = np.full((3, 1), -0.2)
ax.text(
    *(projected_point + offset).flat,
    "({}, {}, {})".format(*np.round(projected_point.flat, 2)),
    color="C0",
    horizontalalignment="right",
)

# add dashed arrow showing projection
arrow_coords = np.concatenate([point, projected_point - point]).flatten()
ax.quiver3D(
    *arrow_coords,
    length=0.96,
    arrow_length_ratio=0.1,
    color="C1",
    linewidth=1,
    linestyle="dashed",
)

# SSP - signal space projection
# Projectors
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(tmax=60).load_data() 
mags = raw.copy().crop(tmax=2).pick(picks="mag")
for proj in (False, True):
    with mne.viz.use_browser_backend("matplotlib"):
        fig = mags.plot(butterfly=True, proj=proj)
    fig.subplots_adjust(top=0.9)
    fig.suptitle(f"proj={proj}", size="xx-large", weight="bold")

plt.show()
