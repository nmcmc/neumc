# This is the modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.
import numpy as np
from matplotlib.patches import RegularPolygon


def add_arrow(ax, xy, direction, radius, color, **kwargs):
    orientation = None
    match direction:
        case "up":
            orientation = 0
        case "right":
            orientation = -np.pi / 2
        case "down":
            orientation = -np.pi
        case "left":
            orientation = np.pi / 2

    ax.add_patch(
        RegularPolygon(
            xy, 3, radius=0.025, orientation=orientation, color=color, **kwargs
        )
    )


def line(ax, ri, rf, *, color="black", **kwargs):
    ax.plot([ri[0], rf[0]], [ri[1], rf[1]], color=color, **kwargs)


def oriented_link(
    ax,
    x,
    y,
    direction,
    *,
    cont_i=False,
    cont_f=False,
    color="black",
    radius=0.025,
    l=0.035,
    d=0.05,
):
    r = np.array((x, y))
    dd_i = None
    disp = None
    dir = None
    match direction:
        case "u":
            disp = np.array((0, 1))
            dir = "up"
            dd_i = (1, 1)
        case "r":
            disp = np.array((1, 0))
            dir = "right"
            dd_i = (1, -1)
        case "d":
            disp = np.array((0, -1))
            dir = "down"
            dd_i = (-1, -1)
        case "l":
            disp = np.array((-1, 0))
            dir = "left"
            dd_i = (-1, 1)

    dd_i = np.asarray(dd_i) * d
    dd_f = dd_i - 2 * disp * d

    r_i = r + dd_i
    r_f = r + disp + dd_f
    r_c = (r_i + r_f) / 2

    r_i = r_i - cont_i * disp * d
    r_f = r_f + cont_f * disp * d

    line(ax, r_i, r_c - disp * l, color=color)
    add_arrow(ax, r_c, dir, radius=radius, color=color, zorder=10)
    line(ax, r_c + disp * l, r_f, color=color)


def add_link_annotation(ax, x, y, direction, d=0.1, fontsize=12):
    xl = x + d
    yd = y + d
    xr = x + 1 - d
    yu = y + 1 - d
    match direction:
        case "up":
            ax.annotate(
                f"$U_{1}({x},{y})$",
                (xl, 0.5 + y),
                (xl + d / 2, 0.5 + y),
                fontsize=12,
                verticalalignment="center",
                horizontalalignment="left",
            )
        case "right":
            ax.annotate(
                f"$U_{0}({x},{y + 1})$",
                (0.5 + x, yu),
                (0.5 + x, yu - d / 2),
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="center",
            )
        case "down":
            ax.annotate(
                f"$U^\\dagger_{1}({x + 1},{y})$",
                (xr, 0.5 + y),
                (xr - d / 2, 0.5 + y),
                fontsize=12,
                verticalalignment="center",
                horizontalalignment="right",
            )
        case "left":
            ax.annotate(
                f"$U^\\dagger_{0}({x},{y})$",
                (0.5 + x, yd),
                (0.5 + x, yd + d / 2),
                fontsize=12,
                verticalalignment="bottom",
                horizontalalignment="center",
            )


def loop(ax, x, y, path, *, d=0.1, l=0.03, r=0.025, color="black", text=None):
    r = np.array((x, y))

    prev_dir = None
    n = len(path)

    for i, direction in enumerate(path):
        cont_i = 0
        cont_f = 0
        next_dir = None
        if i < n - 1:
            next_dir = path[i + 1]

        match direction:
            case "u":
                disp = np.array((0, 1))
                dd = (1, 1)
                if prev_dir == "u":
                    cont_i = 1
                elif prev_dir == "r":
                    cont_i = 2
                if next_dir == "u":
                    cont_f = 1
                if next_dir == "l":
                    cont_f = 2
            case "r":
                disp = np.array((1, 0))
                dd = (1, -1)
                if prev_dir == "r":
                    cont_i = 1
                elif prev_dir == "d":
                    cont_i = 2
                if next_dir == "r":
                    cont_f = 1
                elif next_dir == "u":
                    cont_f = 2
            case "d":
                disp = np.array((0, -1))
                dd = (-1, -1)
                if prev_dir == "d":
                    cont_i = 1
                elif prev_dir == "l":
                    cont_i = 2
                if next_dir == "d":
                    cont_f = 1
                elif next_dir == "r":
                    cont_f = 2
            case "l":
                disp = np.array((-1, 0))
                dd = (-1, 1)
                if prev_dir == "l":
                    cont_i = 1
                elif prev_dir == "u":
                    cont_i = 2
                if next_dir == "l":
                    cont_f = 1
                elif next_dir == "d":
                    cont_f = 2

        dd = np.asarray(dd) * d

        oriented_link(
            ax,
            r[0],
            r[1],
            direction,
            color=color,
            radius=r,
            l=l,
            d=d,
            cont_i=cont_i,
            cont_f=cont_f,
        )
        r += disp
        prev_dir = direction


def plaquette(ax, x, y, *, d=0.1, l=0.35, r=0.025, color="black", text=None):
    xl = x + d
    yd = y + d
    xr = x + 1 - d
    yu = y + 1 - d
    ax.plot([xl, xl, xl + l], [yd + l, yd, yd], color=color)
    ax.plot([xr - l, xr, xr], [yd, yd, yd + l], color=color)
    ax.plot([xr, xr, xr - l], [yu - l, yu, yu], color=color)
    ax.plot([xl + l, xl, xl], [yu, yu, yu - l], color=color)
    ax.add_patch(
        RegularPolygon(
            (0.5 + x, yd), 3, radius=0.025, orientation=np.pi / 2, color=color
        )
    )
    ax.add_patch(
        RegularPolygon((xl, 0.5 + y), 3, radius=0.025, orientation=0, color=color)
    )
    ax.add_patch(
        RegularPolygon(
            (0.5 + x, yu), 3, radius=0.025, orientation=-np.pi / 2, color=color
        )
    )
    ax.add_patch(
        RegularPolygon((xr, 0.5 + y), 3, radius=0.025, orientation=-np.pi, color=color)
    )
    if text is not None:
        ax.text(
            0.5 + x,
            0.5 + y,
            text,
            fontsize=11,
            verticalalignment="center",
            horizontalalignment="center",
            color=color,
        )


def annotated_plaquette(ax, x, y, *, d=0.1, l=0.35, r=0.025, color="black"):
    xl = x + d
    yd = y + d
    xr = x + 1 - d
    yu = y + 1 - d
    ax.plot([xl, xl, xl + l], [yd + l, yd, yd], color=color)
    ax.plot([xr - l, xr, xr], [yd, yd, yd + l], color=color)
    ax.plot([xr, xr, xr - l], [yu - l, yu, yu], color=color)
    ax.plot([xl + l, xl, xl], [yu, yu, yu - l], color=color)
    ax.annotate(
        f"({x},{y})",
        (xl, yd),
        (xl, yd - d / 2),
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="center",
    )

    # ax.add_patch(RegularPolygon((xl, 0.5 + y), 3, radius=0.025, orientation=0, color=color))
    add_arrow(ax, (xl, 0.5 + y), "up", radius=0.025, color=color)
    # ax.annotate(f'$U_{1}({x},{y})$', (xl, 0.5 + y), (xl + d / 2, 0.5 + y), fontsize=12, verticalalignment='center')
    add_link_annotation(ax, x, y, "up", d=d, fontsize=12)

    add_arrow(ax, (0.5 + x, yu), "right", radius=0.025, color=color)
    add_link_annotation(ax, x, y, "right", d=d, fontsize=12)

    add_arrow(ax, (xr, 0.5 + y), "down", radius=0.025, color=color)
    add_link_annotation(ax, x, y, "down", d=d, fontsize=12)

    add_arrow(ax, (0.5 + x, yd), "left", radius=0.025, color=color)
    add_link_annotation(ax, x, y, "left", d=d, fontsize=12)


def link(ax, x, y, mu, *, color, d=0.05, **kwargs):
    if mu == 1:
        ax.plot([x, x], [y + d, y + 1 - d], color=color, **kwargs)
    else:
        ax.plot([x + d, x + 1 - d], [y, y], color=color, **kwargs)


colors = {"active": "orange", "frozen": "green", "passive": "magenta"}


def plot_plaq_mask(ax, mask):
    plaq_mask = mask[1][0]
    link_mask = mask[0]

    L = plaq_mask["active"].shape[0]
    ax.set_axis_off()
    ax.set_xlim(-0.1, 8.1)
    ax.set_ylim(-0.1, 8.1)
    ax.set_aspect(1.0)
    for i in range(L):
        for j in range(L):
            color = "black"
            for k in plaq_mask.keys():
                if plaq_mask[k][i, j] == 1:
                    color = colors[k]
            plaquette(ax, i, j, color=color)

            for mu in (0, 1):
                if link_mask[mu, i, j] == 1:
                    link(ax, i, j, mu, color="blue")
    return ax


from matplotlib.patches import Circle


def plot_scalar_mask(ax, mask):
    keys = list(mask.keys())
    L0, L1 = mask[keys[0]].shape
    ax.set_axis_off()
    ax.set_xlim(-0.25, L1 - 1 + 0.25)
    ax.set_ylim(-0.25, L0 - 1 + 0.25)

    ax.set_aspect(1.0)
    for i in range(L0):
        for j in range(L1):
            color = "black"
            for k in keys:
                if mask[k][i, j] == 1:
                    color = colors[k]
                ax.add_patch(
                    Circle((j, L0 - 1 - i), 0.2, edgecolor=None, facecolor=color)
                )


def plot_grid(ax, lx, ly, *, lines=None, points=None):
    for i in range(Lx):
        pass
    pass
