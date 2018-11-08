import ternary

def draw_pdf_contours(dist, ax):


    axes_colors = {'b': 'g', 'l': 'r', 'r': 'b'}

    scale = 1
    tax = ternary.TernaryAxesSubplot(ax=ax)
    tax.boundary(linewidth=2.0, axes_colors=axes_colors)

    tax.gridlines(multiple=0.1, linewidth=1,
                  horizontal_kwargs={'color': axes_colors['b']},
                  left_kwargs={'color': axes_colors['l']},
                  right_kwargs={'color': axes_colors['r']},
                  alpha=0.7)

    # Plot a few different styles with a legend
    tax.scatter(dist, s=20, marker='o', facecolors=None,
                edgecolors='red')

    fontsize = 10
    tax.right_axis_label("Task 2 kernel $\mathbf{p}$\n", fontsize=fontsize, color=axes_colors['r'], offset=0.15)
    tax.left_axis_label("Task 1 kernel $\mathbf{p}$\n", fontsize=fontsize, color=axes_colors['l'], offset=0.15)
    tax.bottom_axis_label("\nShared kernel $\mathbf{p}$", fontsize=fontsize, color=axes_colors['b'], offset=0.25)

    #tax.right_corner_label('test', fontsize=fontsize, offset=0.10)
    #tax.left_corner_label('test2', fontsize=fontsize, offset=0.10)
    #tax.top_corner_label('Shared $\mathbf{p}=1$', fontsize=fontsize, offset=0.10)

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    # Set and format axes ticks.
    scale = 10
    ticks = [i / float(scale) for i in range(scale + 1)]
    tax.ticks(ticks=ticks, axis='rlb', linewidth=0.7, clockwise=False,
              axes_colors=axes_colors, offset=0.04, tick_formats="%0.1f")