import ternary
import numpy as np

def draw_trajectory(list_of_dist, ax):

    tax = ternary.TernaryAxesSubplot(ax=ax)
    axes_colors = {'b': 'g', 'l': 'r', 'r': 'b'}
    tax.boundary(linewidth=2.0, axes_colors=axes_colors)

    tax.gridlines(multiple=0.1, linewidth=1,
                  horizontal_kwargs={'color': axes_colors['b']},
                  left_kwargs={'color': axes_colors['l']},
                  right_kwargs={'color': axes_colors['r']},
                  alpha=0.7)

    # iterate over kernel
    for kernel_traj in list_of_dist:
        tax.plot_colored_trajectory(kernel_traj, linewidth=1.5)

    fontsize = 10
    tax.right_axis_label("Task 2 kernel $\mathbf{p}$\n", fontsize=fontsize, color=axes_colors['r'], offset=0.15)
    tax.left_axis_label("Task 1 kernel $\mathbf{p}$\n", fontsize=fontsize, color=axes_colors['l'], offset=0.15)
    tax.bottom_axis_label("\nShared kernel $\mathbf{p}$", fontsize=fontsize, color=axes_colors['b'], offset=0.25)

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    # Set and format axes ticks.
    scale = 10
    ticks = [i / float(scale) for i in range(scale + 1)]
    tax.ticks(ticks=ticks, axis='rlb', linewidth=0.7, clockwise=False,
              axes_colors=axes_colors, offset=0.04, tick_formats="%0.1f")


def draw_heat_contours(dist, ax, sigma=None, scale=None):

    if scale is None:
        nbins = 11
        scale = 10
    else:
        nbins = scale + 1

    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

    axes_colors = {'b': 'g', 'l': 'r', 'r': 'b'}
    tax.boundary(linewidth=2.0, axes_colors=axes_colors)

    x = [p[1] for p in dist]
    y = [p[0] for p in dist]
    z = [p[2] for p in dist]
    xyz = np.array([x, y, z]).T

    # 3D histogram
    # xyz is a list of [N, 3] points
    H, b = np.histogramdd((xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                          bins=(nbins, nbins, nbins), range=((0, 1), (0, 1), (0, 1)))
    H = H / np.sum(H)

    # 3D smoothing and interpolation
    from scipy.ndimage.filters import gaussian_filter
    if sigma is None:
        sigma = 5
    kde = gaussian_filter(H, sigma=5)
    interp_dict = dict()
    binx = np.linspace(0, 1, nbins)
    for i, x in enumerate(binx):
        for j, y in enumerate(binx):
            for k, z in enumerate(binx):
                interp_dict[(i, j, k)] = kde[i, j, k]

    tax.heatmap(interp_dict, vmin=np.min(kde), vmax=np.max(kde), colorbar=False)

    fontsize = 10
    tax.right_axis_label("Task 2 kernel $\mathbf{p}$", fontsize=fontsize, color=axes_colors['r'], offset=0.175)
    tax.left_axis_label("Task 1 kernel $\mathbf{p}$", fontsize=fontsize, color=axes_colors['l'], offset=0.175)
    tax.bottom_axis_label("Shared kernel $\mathbf{p}$", fontsize=fontsize, color=axes_colors['b'], offset=0.175)
    tax._redraw_labels()

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    # Set and format axes ticks.
    ticks = [i / 10 for i in range(11)]
    tax.ticks(ticks=ticks, axis='rlb', linewidth=0.7, clockwise=False,
              axes_colors=axes_colors, offset=0.04, tick_formats="%0.1f")


def draw_pdf_contours(dist, ax, is_permute=False):

    axes_colors = {'b': 'g', 'l': 'r', 'r': 'b'}

    tax = ternary.TernaryAxesSubplot(ax=ax)
    tax.boundary(linewidth=2.0, axes_colors=axes_colors)

    tax.gridlines(multiple=0.1, linewidth=1,
                  horizontal_kwargs={'color': axes_colors['b']},
                  left_kwargs={'color': axes_colors['l']},
                  right_kwargs={'color': axes_colors['r']},
                  alpha=0.7)

    if is_permute:
        dist = dist[:, [1, 0, 2]]

    # Plot a few different styles with a legend
    tax.scatter(dist, s=20, marker='o', facecolors=None,
                edgecolors='red')

    fontsize = 10
    tax.right_axis_label("Task 2 kernel $\mathbf{p}$\n", fontsize=fontsize, color=axes_colors['r'], offset=0.125)
    tax.left_axis_label("Task 1 kernel $\mathbf{p}$\n", fontsize=fontsize, color=axes_colors['l'], offset=0.125)
    tax.bottom_axis_label("\nShared kernel $\mathbf{p}$", fontsize=fontsize, color=axes_colors['b'], offset=0.125)

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    # Set and format axes ticks.
    scale = 10
    ticks = [i / float(scale) for i in range(scale + 1)]
    tax.ticks(ticks=ticks, axis='rlb', linewidth=0.7, clockwise=False,
              axes_colors=axes_colors, offset=0.04, tick_formats="%0.1f")