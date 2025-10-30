from typing import Set
import numpy as np
import pandas as pd
import colorsys
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Plot - Chord Diagram Plotly
# -----------------------------------------------------------------------------
# ---------- Prep data ----------
def build_lr_df(df: pd.DataFrame) -> pd.DataFrame:

    df_new = df.copy()[["ligand", "receptor"]]

    # Sort the output table to get ready for chord diagram
    order1 = df_new.ligand.value_counts().index.values
    order2 = df_new.receptor.value_counts().index.values

    df_new['ligand'] = pd.Categorical(df_new['ligand'], categories=order1, ordered=True)
    df_new['receptor'] = pd.Categorical(df_new['receptor'], categories=order2, ordered=True)

    # Sort out receptor order within each cell type first
    df_new = df_new.sort_values(['receptor'])
    df_new['receptor_seq'] = df_new.groupby(['receptor'], observed=False).cumcount()

    # Then sort out ligand order
    df_new = df_new.sort_values(['ligand'])
    df_new['ligand_seq'] = df_new.groupby(['ligand'], observed=False).cumcount()

    df_new['ligand_labeled'] = df_new['ligand']
    df_new['receptor_labeled'] = df_new['receptor'].str.replace("_","/<br>")

    # Generate summary dict
    summary_dict = {}
    ligand_counts_dict = df_new.ligand.value_counts().to_dict()
    receiver_counts_dict = df_new.receptor.value_counts().to_dict()
    summary_dict.update(ligand_counts_dict)
    summary_dict.update(receiver_counts_dict)

    return df_new, summary_dict


def build_chord_diagram_input(all_lr_df_sorted, summary_dict):
    # Order ligands & receptors by frequency
    lig_order = all_lr_df_sorted["ligand"].value_counts().index.tolist()
    rec_order = all_lr_df_sorted["receptor"].value_counts().index.tolist()

    # Make clean display labels, preserving the (l)/(r) suffixes
    # (take the first seen label for each unique name)
    lig_label_map = (all_lr_df_sorted[["ligand", "ligand_labeled"]]
                     .drop_duplicates("ligand")
                     .set_index("ligand")["ligand_labeled"]
                     .to_dict())
    rec_label_map = (all_lr_df_sorted[["receptor", "receptor_labeled"]]
                     .drop_duplicates("receptor")
                     .set_index("receptor")["receptor_labeled"]
                     .to_dict())

    lig_labels = [lig_label_map[x] for x in lig_order]
    rec_labels = [rec_label_map[x] for x in rec_order]

    labels_full = lig_labels[::-1] + rec_labels[::-1]
    N = len(labels_full)

    # Index map in this final order
    idx = {lbl: i for i, lbl in enumerate(labels_full)}

    # Build M_full by iterating interactions (symmetric, bipartite-only)
    M_full = np.zeros((N, N), dtype=int)

    for _, row in all_lr_df_sorted.iterrows():
        li = idx[lig_label_map[row["ligand"]]]  # ligand label index
        rj = idx[rec_label_map[row["receptor"]]]  # receptor label index
        # add 1 for this interaction (or add a weight if you have one)
        M_full[li, rj] += 1
        M_full[rj, li] += 1  # mirror so rows == cols for the ligand/receptor pair

    L = len(lig_labels)
    R = len(rec_labels)
    groups = [list(range(L)), list(range(L, L+R))]

    return labels_full, M_full, groups


# ---------- Plot helpers ----------
def check_data(data_matrix):
    L, M = data_matrix.shape
    if L != M:
        raise ValueError("Data array must be (n,n)")
    return L


def moduloAB(x, a, b):
    if a >= b:
        raise ValueError("Incorrect interval ends")
    y = (x - a) % (b - a)
    return y + b if y < 0 else y + a


def test_2PI(x):
    return 0 <= x < 2 * np.pi


def make_ideogram_arc(R, phi, a=50):
    if not test_2PI(phi[0]) or not test_2PI(phi[1]):
        phi = [moduloAB(t, 0, 2 * np.pi) for t in phi]
    length = (phi[1] - phi[0]) % (2 * np.pi)
    nr = 5 if length <= np.pi / 4 else int(a * length / np.pi)
    if phi[0] < phi[1]:
        theta = np.linspace(phi[0], phi[1], nr)
    else:
        phi = [moduloAB(t, -np.pi, np.pi) for t in phi]
        theta = np.linspace(phi[0], phi[1], nr)
    return R * np.exp(1j * theta)


def get_ideogram_ends(ideogram_len, gap):
    ideo_ends, left = [], 0.0
    for k in range(len(ideogram_len)):
        right = left + ideogram_len[k]
        ideo_ends.append([left, right])
        left = right + gap
    return ideo_ends


def map_data(data_matrix, row_value, ideogram_length):
    L = data_matrix.shape[0]
    mapped = np.zeros_like(data_matrix, dtype=float)
    for j in range(L):
        mapped[:, j] = ideogram_length * data_matrix[:, j] / row_value
    return mapped


def control_pts(angle, radius):
    if len(angle) != 3: raise ValueError("angle must have len=3")
    b_cplx = np.array([np.exp(1j * angle[k]) for k in range(3)])
    b_cplx[1] = radius * b_cplx[1]
    return list(zip(b_cplx.real, b_cplx.imag))


def ctrl_rib_chords(l, r, radius):
    if len(l) != 2 or len(r) != 2:
        raise ValueError("arc ends must be lists of len 2")
    return [control_pts([l[j], (l[j] + r[j]) / 2, r[j]], radius) for j in range(2)]


def make_q_bezier(b):
    if len(b) != 3:
        raise ValueError("control polygon must have 3 points")
    A, B, C = b
    return f"M {A[0]},{A[1]} Q {B[0]}, {B[1]} {C[0]}, {C[1]}"


def make_ribbon_arc(theta0, theta1):
    """
    Build the outer arc as a straight angular sweep from theta0 to theta1
    in their given numeric order.
    This matches how ribbon_ends were constructed (cumulative, monotone).
    Returns an SVG 'L x,y' polyline string.
    """
    d = float(theta1 - theta0)
    n = max(3, int(40 * abs(d) / np.pi))  # density ~ arc length
    theta = np.linspace(theta0, theta1, n)
    z = np.exp(1j * (theta % (2 * np.pi)))
    return "".join(f"L {z.real[k]}, {z.imag[k]} " for k in range(n))


def make_layout(title, plot_size):
    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title="")
    return go.Layout(
        title=title,
        xaxis=dict(**axis), yaxis=dict(**axis),
        showlegend=False, width=plot_size, height=plot_size,
        margin=dict(t=25, b=25, l=25, r=25),
        hovermode="closest"
    )


def make_ideo_shape(path, line_color, fill_color):
    return dict(
        line=dict(color=line_color, width=0.45),
        path=path, type="path", fillcolor=fill_color, layer="below"
    )


def make_ribbon(l, r, line_color, fill_color, radius=0.2):
    polygon = ctrl_rib_chords(l, r, radius)
    b, c = polygon
    return dict(
        line=dict(color=line_color, width=0),
        path=make_q_bezier(b) + make_ribbon_arc(r[0], r[1]) + make_q_bezier(c[::-1]) + make_ribbon_arc(l[1], l[0]),
        type="path", fillcolor=fill_color, layer="below"
    )


def make_self_rel(l, line_color, fill_color, radius):
    b = control_pts([l[0], (l[0] + l[1]) / 2, l[1]], radius)
    return dict(
        line=dict(color=line_color, width=0),
        path=make_q_bezier(b) + make_ribbon_arc(l[1], l[0]),
        type="path", fillcolor=fill_color, layer="below"
    )


def invPerm(perm):
    inv = [0] * len(perm)
    for i, s in enumerate(perm):
        inv[s] = i
    return inv


def make_colors(L, alpha=0.75, s=0.55, v=0.95,
                start_offset=0.5,  # 0.5 = start at 9 o’clock
                clockwise=True):
    cols = []
    for k in range(L):
        # map k→hue with offset, reverse direction for clockwise
        frac = k / L
        h = (start_offset - frac) % 1.0 if clockwise else (start_offset + frac) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append(f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})")
    return cols


def _add_group_ring(fig, ideo_ends, groups_idx, group_label, group_color,
                    r_inner, r_outer, label_size=12, label_color="black",
                    label_gap=0.05):
    group_shapes, group_annots = [], []
    for g, name, col in zip(groups_idx, group_label, group_color):
        # contiguous block from min-start to max-end
        start = min(ideo_ends[i][0] for i in g)
        end = max(ideo_ends[i][1] for i in g)

        # arcs
        z_outer = make_ideogram_arc(r_outer, [start, end])
        z_inner = make_ideogram_arc(r_inner, [start, end])

        path = "M " + " ".join(f"{z_outer.real[k]}, {z_outer.imag[k]} L "
                               for k in range(len(z_outer)))
        Zi = np.array(z_inner.tolist()[::-1])
        path += " ".join(f"{Zi.real[k]}, {Zi.imag[k]} L " for k in range(len(Zi)))
        path += f"{z_outer.real[0]}, {z_outer.imag[0]}"

        group_shapes.append(dict(
            type="path",
            path=path,
            fillcolor=col,
            layer="below",
            line=dict(color="rgba(0,0,0,0)", width=0)  # no border
        ))

        # label at mid-angle, horizontal
        mid = 0.5 * (start + end)
        r_lab = r_outer + label_gap
        xg, yg = r_lab * np.cos(mid), r_lab * np.sin(mid)

        group_annots.append(dict(
            x=xg, y=yg, xref="x", yref="y",
            text=name, showarrow=False,
            font=dict(size=label_size, color=label_color),
            xanchor="center", yanchor="middle",
            textangle=0
        ))

    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    fig.update_layout(shapes=existing_shapes + group_shapes)
    existing_ann = list(fig.layout.annotations) if fig.layout.annotations else []
    fig.update_layout(annotations=existing_ann + group_annots)


# ---------- Main plot ----------
def chord_diagram(labels, matrix, title="Chord diagram",
                  plot_size=800, gap_frac=0.005, ideo_outer=1.1, ideo_inner=1.0,
                  colors=None,
                  groups=None,
                  group_gaps_deg=5.0,
                  rotate_deg=0.0,
                  label_radius_factor=1.18,
                  label_font_size=10,
                  group_ring=False,
                  group_indices=None,  # list[list[int]] in final label order
                  group_label=None,  # list[str]
                  group_color=None,  # list[str] (hex/rgba)
                  group_ring_gap=0.05,  # data-space gap from label ring
                  group_ring_width=0.05,  # thickness of the group band
                  group_label_size=12,
                  group_label_color="black"):
    matrix = np.array(matrix, dtype=int)
    L = check_data(matrix)

    row_sum = [np.sum(matrix[k, :]) for k in range(L)]
    if sum(row_sum) == 0:
        raise ValueError("All rows are zero.")

    weights = np.asarray(row_sum, dtype=float)
    W = weights.sum()
    L = len(labels)
    PI2 = 2 * np.pi

    # base per-sector gap (capped globally)
    desired_per_gap = PI2 * gap_frac
    max_total_gap = PI2 * 0.2
    base_total_gap = min(desired_per_gap * L, max_total_gap)
    per_sector_gap = base_total_gap / max(1, L)  # small gap between every adjacent pair

    # normalize groups to index-lists in the *current* label order
    def canon_groups(groups, labels):
        if groups is None:
            return None
        name_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        canon = []
        seen = set()
        for g in groups:
            gi = []
            for item in g:
                idx = int(item) if isinstance(item, (int, np.integer)) else name_to_idx[item]
                if idx in seen:
                    raise ValueError(f"Label/index used in multiple groups: {labels[idx]}")
                seen.add(idx)
                gi.append(idx)
            canon.append(sorted(gi, key=lambda x: x))  # keep index order
        # if not all sectors are covered, the uncovered remainder is a trailing group
        if len(seen) < L:
            canon.append([i for i in range(L) if i not in seen])
        return canon

    canon = canon_groups(groups, labels)

    # build a per-boundary gap array, with extra gap at each group boundary (wrap included)
    # gap_after[i] is the gap placed after sector i (between i and (i+1)%L)
    gap_after = np.full(L, per_sector_gap, dtype=float)

    if canon and len(canon) > 0:
        # map each index to its group id (following the current order 0..L-1)
        group_id = np.full(L, -1, dtype=int)
        for gid, g in enumerate(canon):
            for idx in g:
                group_id[idx] = gid
        if np.any(group_id < 0):
            raise ValueError("Some sectors not assigned to any group")

        # detect boundaries: where group changes between i and (i+1)%L
        boundaries = [i for i in range(L) if group_id[i] != group_id[(i + 1) % L]]

        # normalize group_gaps_deg into radians (either scalar or per-boundary list)
        if isinstance(group_gaps_deg, (int, float)):
            extra_list = [np.deg2rad(group_gaps_deg)] * len(boundaries)
        else:
            if len(group_gaps_deg) != len(boundaries):
                raise ValueError("group_gaps_deg must have one entry per detected boundary")
            extra_list = [np.deg2rad(float(x)) for x in group_gaps_deg]

        for b_idx, i in enumerate(boundaries):
            gap_after[i] += extra_list[b_idx]

    # compute available angle for ideograms after reserving all gaps
    total_gap = float(gap_after.sum())
    available = PI2 - total_gap
    if available <= 0:
        raise ValueError("Gaps are too large; no room left for sectors. Reduce gap_frac/group_gaps.")

    # proportional spans with a small minimum width, rescaled to 'available'
    spans = available * (weights / W)

    min_span = PI2 * 0.0008
    too_small = spans < min_span
    if np.any(too_small):
        n_small = int(too_small.sum())
        spans[too_small] = min_span
        remaining = available - n_small * min_span
        mask = ~too_small
        if mask.any() and remaining > 0:
            spans[mask] *= remaining / spans[mask].sum()

    # integrate: start/end for each sector, adding gap_after[i] after each sector
    offset = np.deg2rad(float(rotate_deg))

    # integrate with rotation applied
    starts = np.zeros(L, dtype=float)
    ends = np.zeros(L, dtype=float)
    theta = offset  # start from rotated angle
    for i in range(L):
        starts[i] = theta
        ends[i] = theta + spans[i]
        theta = ends[i] + gap_after[i]

    # final ideogram ends as [[start_i, end_i] ...]
    ideo_ends = [[float(starts[i]), float(ends[i])] for i in range(L)]

    mapped = map_data(matrix, row_sum, spans)
    idx_sort = np.argsort(mapped, axis=1)

    def make_ribbon_ends(mapped_data, ideo_ends, idx_sort):
        L = mapped_data.shape[0]
        ribbon_boundary = np.zeros((L, L+1))
        for k in range(L):
            start = ideo_ends[k][0]
            ribbon_boundary[k][0] = start
            for j in range(1, L+1):
                J = idx_sort[k][j-1]
                ribbon_boundary[k][j] = start + mapped_data[k][J]
                start = ribbon_boundary[k][j]
        return [[(ribbon_boundary[k][j], ribbon_boundary[k][j+1]) for j in range(L)]
                for k in range(L)]

    ribbon_ends = make_ribbon_ends(mapped, ideo_ends, idx_sort)

    # Colors: generate enough for all groups
    ideo_colors = colors if colors is not None else make_colors(L, alpha=0.75)
    if len(ideo_colors) < L:
        raise ValueError("Not enough colors provided for all groups.")
    ribbon_color = [L * [ideo_colors[k]] for k in range(L)]

    layout = make_layout(title, plot_size)
    shapes = []
    hover_markers = []
    radii_sribb = [0.35] * L

    # Ribbons
    for k in range(L):
        sigma = idx_sort[k]
        sigma_inv = invPerm(sigma)
        for j in range(k, L):
            if matrix[k][j] == 0 and matrix[j][k] == 0:
                continue
            eta = idx_sort[j]
            eta_inv = invPerm(eta)
            l = ribbon_ends[k][sigma_inv[j]]

            if j == k:
                shapes.append(make_self_rel(l, "rgba(0,0,0,0)", ideo_colors[k], radius=radii_sribb[k]))
                z = 0.97 * np.exp(1j * (l[0] + l[1]) / 2)
                hover_markers.append(go.Scatter(
                    x=[z.real], y=[z.imag], mode="markers",
                    marker=dict(size=0.5, color=ideo_colors[k]),
                    text=f"{labels[k]} ↔ {labels[k]}: {matrix[k][k]}",
                    hoverinfo="text", showlegend=False
                ))
            else:
                r = ribbon_ends[j][eta_inv[k]]
                zi = 0.97 * np.exp(1j * (l[0] + l[1]) / 2)
                zf = 0.97 * np.exp(1j * (r[0] + r[1]) / 2)
                hover_markers += [
                    go.Scatter(x=[zi.real], y=[zi.imag], mode="markers",
                               marker=dict(size=0.5, color=ribbon_color[k][j]),
                               text=f"{labels[k]} → {labels[j]}: {matrix[k][j]}",
                               hoverinfo="text", showlegend=False),
                    go.Scatter(x=[zf.real], y=[zf.imag], mode="markers",
                               marker=dict(size=0.5, color=ribbon_color[k][j]),
                               text=f"{labels[k]} → {labels[j]}: {matrix[j][k]}",
                               hoverinfo="text", showlegend=False),
                ]
                r = (r[1], r[0])  # reverse for second Bezier
                shapes.append(make_ribbon(l, r, "rgba(0,0,0,0)", ribbon_color[k][j]))

    # Ideograms
    ideograms = []
    for k in range(len(ideo_ends)):
        z = make_ideogram_arc(ideo_outer, ideo_ends[k])
        zi = make_ideogram_arc(ideo_inner, ideo_ends[k])
        m = len(z)

        ideograms.append(go.Scatter(
            x=z.real, y=z.imag, mode="lines",
            line=dict(color=ideo_colors[k], shape="spline", width=1),
            text=f"{labels[k]}<br>{int(row_sum[k])}",
            hoverinfo="text", showlegend=False
        ))

        path = "M " + " ".join(f"{z.real[s]}, {z.imag[s]} L " for s in range(m))
        Zi = np.array(zi.tolist()[::-1])
        path += " ".join(f"{Zi.real[s]}, {Zi.imag[s]} L " for s in range(m))
        path += f"{z.real[0]} ,{z.imag[0]}"
        shapes.append(make_ideo_shape(path, "black", ideo_colors[k]))

    # Radial (perpendicular-to-arc) labels outside the ring
    label_annotations = []
    r_lab = float(ideo_outer) * label_radius_factor  # how far outside the ring

    for k in range(len(ideo_ends)):
        start = ideo_ends[k][0]
        end = ideo_ends[k][1]
        mid = (start + end) * 0.5  # sector midpoint angle (radians)
        x = r_lab * np.cos(mid)
        y = r_lab * np.sin(mid)

        # Base radial orientation: angle in degrees (0° = +x axis), CCW positive
        ang = (-np.degrees(mid)) % 360

        # Keep text upright: flip if angle is upside-down (90..270°)
        readable_ang = ang + 180 if 90 < ang < 270 else ang

        # Anchor so text grows outward from the circle
        label_annotations.append(dict(
            x=x, y=y,
            text=labels[k],
            showarrow=False,
            font=dict(size=label_font_size, color="black"),
            textangle=readable_ang,
            xanchor="center",
            yanchor="middle",
            align="center"
        ))

    fig = go.Figure(data=ideograms + hover_markers, layout=layout)

    pad = ideo_outer * float(label_radius_factor) + 0.5
    fig.update_layout(title=dict(
                          text=title,
                          x=0.5,
                          xanchor="center",
                          yanchor="top",
                          #y=0.95,
                          font=dict(size=16, color="black", family="Arial")
                      ),
                      shapes=shapes,
                      annotations=label_annotations,
                      plot_bgcolor="white",
                      paper_bgcolor="white")
    fig.update_xaxes(visible=False, range=[-pad, pad])
    fig.update_yaxes(visible=False, range=[-pad, pad],
                     scaleanchor="x", scaleratio=1)

    # optional outer group ring
    if group_ring:
        if not (group_indices and group_label and group_color):
            raise ValueError("When group_ring=True, provide group_indices, group_label, and group_color.")

        # ring radii just outside label ring
        r_label = ideo_outer * float(label_radius_factor)
        r_inner = r_label + float(group_ring_gap)
        r_outer = r_inner + float(group_ring_width)

        _add_group_ring(
            fig, ideo_ends,
            groups_idx=group_indices,
            group_label=group_label,
            group_color=group_color,
            r_inner=r_inner, r_outer=r_outer,
            label_size=group_label_size,
            label_color=group_label_color
        )

    return fig