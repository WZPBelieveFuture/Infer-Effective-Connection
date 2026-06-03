import argparse
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch


ROI_LABELS = ["PCC", "LACC", "LMTG", "LAG", "RACC", "RMTG", "RAG"]
ROI_POSITIONS = {
    "LMTG": (0.33, 0.26),
    "LACC": (0.50, 0.40),
    "LAG": (0.36, 0.79),
    "PCC": (0.54, 0.77),
    "RACC": (0.66, 0.40),
    "RMTG": (0.82, 0.26),
    "RAG": (0.79, 0.79),
}
ROI_COLORS = {
    "PCC": "#f59e0b",
    "LACC": "#ef4444",
    "LMTG": "#f97316",
    "LAG": "#fb7185",
    "RACC": "#06b6d4",
    "RMTG": "#3b82f6",
    "RAG": "#14b8a6",
}


def build_macro_lorenz96_adjacency(num_nodes):
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        adjacency[i, i] = 1.0
        adjacency[i, (i - 1) % num_nodes] = 1.0
        adjacency[i, (i - 2) % num_nodes] = 1.0
        adjacency[i, (i + 1) % num_nodes] = 1.0
    return adjacency


def load_square_matrix_csv(matrix_path, matrix_name):
    matrix_path = Path(matrix_path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"{matrix_name} file not found: {matrix_path}")

    candidate_frames = [
        pd.read_csv(matrix_path),
        pd.read_csv(matrix_path, header=None),
    ]
    for frame in candidate_frames:
        values = frame.values.astype(np.float32)
        if values.ndim == 2 and values.shape[0] == values.shape[1]:
            return values

    shapes = ", ".join(str(frame.values.shape) for frame in candidate_frames)
    raise ValueError(f"{matrix_name} must be a square matrix CSV. Tried shapes: {shapes}")


def compute_auc_score(labels, scores):
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return np.nan

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(scores.size, dtype=np.float64)

    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end

    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = sorted_ranks
    positive_rank_sum = float(ranks[labels].sum())
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def compute_threshold_metrics(score_matrix, ground_truth):
    scores = np.asarray(score_matrix, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32) > 0
    if scores.shape != ground_truth.shape:
        raise ValueError(
            f"score matrix shape {scores.shape} does not match ground truth shape {ground_truth.shape}"
        )

    auc = compute_auc_score(ground_truth, scores)
    thresholds = np.unique(scores.reshape(-1))
    best_metrics = None
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.logical_and(predicted, ground_truth).sum())
        tn = int(np.logical_and(~predicted, ~ground_truth).sum())
        fp = int(np.logical_and(predicted, ~ground_truth).sum())
        fn = int(np.logical_and(~predicted, ground_truth).sum())

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        candidate = {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": auc,
        }
        if best_metrics is None or (candidate["f1"], candidate["accuracy"]) > (
            best_metrics["f1"],
            best_metrics["accuracy"],
        ):
            best_metrics = candidate

    return best_metrics


def format_metric_text(metrics):
    auc = metrics["auc"]
    auc_text = "nan" if np.isnan(auc) else f"{auc:.3f}"
    return (
        f"thr={metrics['threshold']:.3g}  Acc={metrics['accuracy']:.3f}\n"
        f"F1={metrics['f1']:.3f}  AUC={auc_text}"
    )


def _heatmap_limits(matrix):
    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def create_lorzen_comparison_figure(
    jacobian_strength,
    ei_causal_graph,
    ground_truth,
    metrics_by_name,
    output_png,
    scale_number,
):
    macro_size = int(ground_truth.shape[0])
    node_labels = [f"M{i + 1}" for i in range(macro_size)]

    fig = plt.figure(figsize=(16, 7.5), dpi=150, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.18, 0.82], width_ratios=[1, 1, 1])

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.72,
        f"Lorzen Causal Graph Comparison (Scale {scale_number})",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#1f2933",
    )
    title_ax.text(
        0.5,
        0.25,
        "Comparing jacobian_mean_abs, ei_causal_graph, and Lorenz-96 groundtruth. Edge direction: source -> target.",
        ha="center",
        va="center",
        fontsize=12,
        color="#52606d",
    )

    heatmap_specs = [
        (
            jacobian_strength,
            "Jacobian Mean Abs",
            "jacobian",
            "YlGnBu",
            "Mean |Jacobian| strength",
            False,
        ),
        (
            ei_causal_graph,
            "EI Causal Graph",
            "ei",
            "YlOrRd",
            "EI causal strength",
            False,
        ),
        (
            ground_truth,
            "Groundtruth Lorenz-96",
            None,
            "Blues",
            "Groundtruth edge",
            True,
        ),
    ]

    for index, (matrix, title, metric_key, cmap, cbar_label, annotate) in enumerate(heatmap_specs):
        ax = fig.add_subplot(gs[1, index])
        vmin, vmax = (0.0, 1.0) if annotate else _heatmap_limits(matrix)
        title_text = title
        if metric_key is not None:
            title_text = f"{title}\n{format_metric_text(metrics_by_name[metric_key])}"
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.8,
            linecolor="white",
            xticklabels=node_labels,
            yticklabels=node_labels,
            annot=annotate,
            fmt=".0f" if annotate else ".2g",
            cbar=True,
            cbar_kws={"shrink": 0.82, "label": cbar_label},
        )
        ax.set_title(title_text, fontsize=13, fontweight="bold", color="#1f2933")
        ax.set_xlabel("Source macro variable", fontsize=10.5)
        ax.set_ylabel("Target macro variable", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_comparison_figure(causal_graph_strength, threshold, metrics, output_png):
    macro_size = causal_graph_strength.shape[0]
    ground_truth = build_macro_lorenz96_adjacency(macro_size)
    node_labels = [f"M{i + 1}" for i in range(macro_size)]
    inferred_heatmap = causal_graph_strength.copy()
    vmax = float(inferred_heatmap.max())

    fig = plt.figure(figsize=(13.333, 7.5), dpi=150, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.18, 0.82], width_ratios=[1, 1])

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.72,
        "Macro Causal Network Comparison",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#1f2933",
    )
    metric_text = (
        f"Threshold = {threshold:.4f}    "
        f"Accuracy = {metrics['causal_accuracy']:.4f}    "
        f"F1 = {metrics['causal_f1']:.4f}    "
        f"AUC = {metrics['auc']:.4f}"
    )
    title_ax.text(
        0.5,
        0.25,
        metric_text,
        ha="center",
        va="center",
        fontsize=12,
        color="#52606d",
    )

    ax_pred = fig.add_subplot(gs[1, 0])
    ax_gt = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        inferred_heatmap,
        ax=ax_pred,
        cmap="YlOrRd",
        vmin=float(inferred_heatmap.min()),
        vmax=vmax,
        square=True,
        linewidths=0.8,
        linecolor="white",
        xticklabels=node_labels,
        yticklabels=node_labels,
        cbar=True,
        cbar_kws={"shrink": 0.85, "label": "EI causal strength"},
    )
    ax_pred.set_title("Inferred Macro Causal Heatmap", fontsize=15, fontweight="bold")
    ax_pred.set_xlabel("Source macro variable", fontsize=11)
    ax_pred.set_ylabel("Target macro variable", fontsize=11)

    sns.heatmap(
        ground_truth,
        ax=ax_gt,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.8,
        linecolor="white",
        xticklabels=node_labels,
        yticklabels=node_labels,
        annot=True,
        fmt=".0f",
        cbar=True,
        cbar_kws={"shrink": 0.85, "label": "Ground-truth edge"},
    )
    ax_gt.set_title("Ground Truth Lorenz-96 Heatmap", fontsize=15, fontweight="bold")
    ax_gt.set_xlabel("Source macro variable", fontsize=11)
    ax_gt.set_ylabel("Target macro variable", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_training_history_figure(history_df, output_png):
    if "ei" not in history_df.columns and "EI" in history_df.columns:
        history_df = history_df.rename(columns={"EI": "ei"})

    required_columns = ["epoch", "train_mse", "val_mse", "pred_loss", "backward_loss"]
    missing_columns = [column for column in required_columns if column not in history_df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in training history: {missing_text}")

    epochs = history_df["epoch"].to_numpy()
    curve_specs = [
        (
            [("train_mse", "Train MSE", "#2563eb"), ("val_mse", "Val MSE", "#dc2626")],
            "Train / Val MSE by Epoch",
            "MSE",
        ),
        (
            [("pred_loss", "Forward Loss", "#059669"), ("backward_loss", "Backward Loss", "#d97706")],
            "Prediction / Backward Loss by Epoch",
            "Loss",
        ),
    ]
    if "ei" in history_df.columns:
        curve_specs.append(
            (
                [("ei", "EI", "#7c3aed")],
                "EI by Epoch",
                "EI",
            )
        )

    fig_height = 10 if len(curve_specs) == 3 else 7.4
    fig, axes = plt.subplots(
        len(curve_specs),
        1,
        figsize=(13.333, fig_height),
        dpi=150,
        facecolor="white",
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (curves, title, ylabel) in zip(axes, curve_specs):
        for column, label, color in curves:
            ax.plot(epochs, history_df[column].to_numpy(), label=label, color=color, linewidth=2.2)
        ax.set_title(title, fontsize=14, fontweight="bold", color="#1f2933")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.legend(frameon=False, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Epoch", fontsize=11)
    fig.suptitle("Training History Overview", fontsize=20, fontweight="bold", color="#1f2933")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_causal_graph(causal_graph_path):
    causal_graph_strength = pd.read_csv(causal_graph_path).values.astype(np.float32)
    if causal_graph_strength.shape != (7, 7):
        raise ValueError(
            f"Expected a 7x7 causal graph for the seven ROIs, got {causal_graph_strength.shape}."
        )
    return causal_graph_strength


def load_jacobian_summary(summary_path):
    if not summary_path.exists():
        return {}
    return pd.read_csv(summary_path).iloc[0].to_dict()


def resolve_saved_scale_number(scale_id):
    # Training uses zero-based logical scale ids, while saved artifacts use one-based file names.
    return int(scale_id) + 1


def build_edge_records(causal_graph_strength):
    records = []
    for target_idx, target_label in enumerate(ROI_LABELS):
        for source_idx, source_label in enumerate(ROI_LABELS):
            if source_idx == target_idx:
                continue
            records.append(
                {
                    "source_label": source_label,
                    "target_label": target_label,
                    "weight": float(causal_graph_strength[target_idx, source_idx]),
                }
            )
    return records


def compute_edge_normalizer(edge_records):
    weights = np.array([record["weight"] for record in edge_records], dtype=np.float32)
    min_weight = float(weights.min())
    max_weight = float(weights.max())
    if np.isclose(min_weight, max_weight):
        max_weight = min_weight + 1e-6
    return Normalize(vmin=min_weight, vmax=max_weight)


def draw_directed_edge(ax, source_xy, target_xy, strength_norm, color):
    width = 0.8 + 4.0 * strength_norm
    alpha = 0.12 + 0.83 * strength_norm
    source_x, _ = source_xy
    target_x, _ = target_xy
    curve_direction = 1 if source_x <= target_x else -1
    radial = 0.18 * curve_direction
    if abs(source_x - target_x) < 0.08:
        radial = 0.12 * curve_direction

    patch = FancyArrowPatch(
        posA=source_xy,
        posB=target_xy,
        arrowstyle="-|>",
        mutation_scale=10 + 8 * strength_norm,
        connectionstyle=f"arc3,rad={radial}",
        linewidth=width,
        color=color,
        alpha=alpha,
        shrinkA=22,
        shrinkB=22,
        zorder=1,
    )
    ax.add_patch(patch)


def create_inferred_network_figure(causal_graph_strength, output_png, source_name, metadata=None):
    metadata = metadata or {}
    edge_records = build_edge_records(causal_graph_strength)
    norm = compute_edge_normalizer(edge_records)
    cmap = get_cmap("YlOrRd")

    fig = plt.figure(figsize=(13.333, 7.5), dpi=150, facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.17, 0.83])

    title_ax = fig.add_subplot(gs[0, 0])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.72,
        "Jacobian-Inferred Causal Network Across 7 ROIs",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#1f2933",
    )
    sample_count = metadata.get("sample_count")
    source_mode = metadata.get("source_mode")
    sample_text = "" if sample_count is None else f"    Samples: {int(sample_count)}"
    mode_text = "" if not source_mode else f"    Mode: {source_mode}"
    title_ax.text(
        0.5,
        0.28,
        f"Source: {source_name}{sample_text}{mode_text}    Self-loops omitted    Edge direction: source -> target",
        ha="center",
        va="center",
        fontsize=12,
        color="#52606d",
    )

    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.02, 0.98)
    ax.set_aspect("equal")
    ax.axis("off")

    for record in edge_records:
        source_xy = ROI_POSITIONS[record["source_label"]]
        target_xy = ROI_POSITIONS[record["target_label"]]
        strength_norm = float(norm(record["weight"]))
        color = cmap(0.25 + 0.7 * strength_norm)
        draw_directed_edge(ax, source_xy, target_xy, strength_norm, color)

    for label in ROI_LABELS:
        x, y = ROI_POSITIONS[label]
        ax.scatter(
            [x],
            [y],
            s=1800,
            color=ROI_COLORS[label],
            edgecolors="white",
            linewidths=2.5,
            zorder=3,
        )
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    off_diagonal_weights = causal_graph_strength[~np.eye(causal_graph_strength.shape[0], dtype=bool)]
    top_edge_records = sorted(edge_records, key=lambda record: record["weight"], reverse=True)[:8]
    top_edge_text = "\n".join(
        [
            f"{index + 1}. {record['source_label']} -> {record['target_label']} ({record['weight']:.3f})"
            for index, record in enumerate(top_edge_records)
        ]
    )
    fig.text(
        0.08,
        0.68,
        (
            "Strongest inferred inter-ROI edges\n"
            f"{top_edge_text}\n\n"
            f"Off-diagonal range: [{float(off_diagonal_weights.min()):.3f}, "
            f"{float(off_diagonal_weights.max()):.3f}]"
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        color="#334155",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
            "linewidth": 1.0,
        },
        zorder=5,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_label("Inferred causal strength", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_minimal_pptx(image_bytes, image_name, output_pptx, deck_title):
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    slide_width = 12192000
    slide_height = 6858000
    deck_title_xml = escape(deck_title)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    app_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
  <PresentationFormat>Widescreen</PresentationFormat>
  <Slides>1</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Slides</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>1</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr">
      <vt:lpstr>{deck_title_xml}</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company>OpenAI</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>1.0</AppVersion>
</Properties>
"""
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{deck_title_xml}</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""
    presentation_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldIdLst>
    <p:sldId id="256" r:id="rId1"/>
  </p:sldIdLst>
  <p:sldSz cx="{slide_width}" cy="{slide_height}"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""
    presentation_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>
"""
    slide_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="Figure"/>
          <p:cNvPicPr/>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="rId1"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="{slide_width}" cy="{slide_height}"/>
          </a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sld>
"""
    slide_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{escape(image_name)}"/>
</Relationships>
"""

    with zipfile.ZipFile(output_pptx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("ppt/presentation.xml", presentation_xml)
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        zf.writestr("ppt/slides/slide1.xml", slide_xml)
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", slide_rels)
        zf.writestr(f"ppt/media/{image_name}", image_bytes)


def run_compare_mode(args):
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_root = Path(args.loc_result_dir) / args.run_name
    saved_scale = resolve_saved_scale_number(args.scale_id)

    jacobian_path = result_root / f"jacobian_mean_abs_scale{saved_scale}.csv"
    causal_graph_path = result_root / f"ei_causal_graph_scale{saved_scale}.csv"
    summary_path = result_root / f"summary_scale{saved_scale}.csv"
    history_path = result_root / f"train_history_scale{saved_scale}.csv"

    jacobian_strength = load_square_matrix_csv(jacobian_path, "jacobian_mean_abs")
    causal_graph_strength = load_square_matrix_csv(causal_graph_path, "ei_causal_graph")
    if jacobian_strength.shape != causal_graph_strength.shape:
        raise ValueError(
            "jacobian_mean_abs and ei_causal_graph must have the same shape. "
            f"Got {jacobian_strength.shape} and {causal_graph_strength.shape}."
        )

    if args.ground_truth_path:
        ground_truth = load_square_matrix_csv(args.ground_truth_path, "ground_truth")
        if ground_truth.shape != causal_graph_strength.shape:
            raise ValueError(
                "ground_truth must have the same shape as the inferred matrices. "
                f"Got {ground_truth.shape} and {causal_graph_strength.shape}."
            )
    else:
        ground_truth = build_macro_lorenz96_adjacency(causal_graph_strength.shape[0])

    history_df = pd.read_csv(history_path)
    metrics_by_name = {
        "jacobian": compute_threshold_metrics(jacobian_strength, ground_truth),
        "ei": compute_threshold_metrics(causal_graph_strength, ground_truth),
    }

    output_png = result_dir / f"lorzen_causal_graph_comparison_scale{saved_scale}.png"
    output_pptx = result_dir / f"lorzen_causal_graph_comparison_scale{saved_scale}.pptx"
    history_png = result_dir / f"training_history_scale{saved_scale}.png"

    create_lorzen_comparison_figure(
        jacobian_strength=jacobian_strength,
        ei_causal_graph=causal_graph_strength,
        ground_truth=ground_truth,
        metrics_by_name=metrics_by_name,
        output_png=output_png,
        scale_number=saved_scale,
    )
    create_training_history_figure(history_df, history_png)
    image_bytes = output_png.read_bytes()
    build_minimal_pptx(image_bytes, output_png.name, output_pptx, "Lorzen Causal Graph Comparison")

    print(f"Saved Lorzen comparison image to {output_png}")
    print(f"Saved PPT to {output_pptx}")
    print(f"Saved training history image to {history_png}")
    print(f"Jacobian metrics: {format_metric_text(metrics_by_name['jacobian'])}")
    print(f"EI metrics: {format_metric_text(metrics_by_name['ei'])}")


def run_real_fmri_roi_mode(args):
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_root = Path(args.loc_result_dir) / args.run_name
    saved_scale = resolve_saved_scale_number(args.scale_id)

    causal_graph_path = result_root / f"jacobian_mean_abs_realdata{args.sample_count_tag}_scale{saved_scale}.csv"
    summary_path = result_root / f"jacobian_mean_abs_realdata{args.sample_count_tag}_scale{saved_scale}_summary.csv"
    causal_graph_strength = load_causal_graph(causal_graph_path)
    metadata = load_jacobian_summary(summary_path)

    output_png = result_dir / f"inferred_causal_network_scale{saved_scale}.png"
    output_pptx = result_dir / f"inferred_causal_network_scale{saved_scale}.pptx"
    deck_title = f"Jacobian-Inferred Causal Network Across 7 ROIs (Scale {saved_scale})"

    create_inferred_network_figure(
        causal_graph_strength,
        output_png,
        causal_graph_path.name,
        metadata=metadata,
    )
    image_bytes = output_png.read_bytes()
    build_minimal_pptx(image_bytes, output_png.name, output_pptx, deck_title)

    print(f"Saved inferred causal network image to {output_png}")
    print(f"Saved PPT to {output_pptx}")


def main():
    parser = argparse.ArgumentParser(
        description="Create PPT figures for Lorenz-style causal comparisons or real-fMRI ROI causal networks."
    )
    parser.add_argument("--mode", type=str, default="compare", choices=["compare", "real_fmri_roi"])
    parser.add_argument("--scale_id", type=int, default=0)
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--loc_result_dir", type=str, default="loc_result_stage2")
    parser.add_argument("--run_name", type=str, default="stage2_lorzen_macro")
    parser.add_argument("--sample_count_tag", type=int, default=2000)
    parser.add_argument("--ground_truth_path", type=str, default="")
    args = parser.parse_args()

    if args.mode == "compare":
        run_compare_mode(args)
        return
    run_real_fmri_roi_mode(args)


if __name__ == "__main__":
    main()
