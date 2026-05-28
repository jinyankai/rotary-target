from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    display_name: str
    family: str
    upstream_config: str
    angle_version: str
    paper_hint: str


MMROTATE_ALGORITHMS: dict[str, AlgorithmSpec] = {
    "roi_trans": AlgorithmSpec(
        key="roi_trans",
        display_name="RoI Transformer",
        family="two_stage",
        upstream_config="roi_trans/roi_trans_r50_fpn_1x_dota_le90.py",
        angle_version="le90",
        paper_hint="CVPR 2019",
    ),
    "gliding_vertex": AlgorithmSpec(
        key="gliding_vertex",
        display_name="Gliding Vertex",
        family="two_stage",
        upstream_config="gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90.py",
        angle_version="le90",
        paper_hint="TPAMI 2021",
    ),
    "redet": AlgorithmSpec(
        key="redet",
        display_name="ReDet",
        family="rotation_equivariant",
        upstream_config="redet/redet_re50_refpn_1x_dota_le90.py",
        angle_version="le90",
        paper_hint="CVPR 2021",
    ),
    "csl": AlgorithmSpec(
        key="csl",
        display_name="Circular Smooth Label",
        family="angle_classification",
        upstream_config="csl/rotated_retinanet_obb_csl_gaussian_r50_fpn_1x_dota_le90.py",
        angle_version="le90",
        paper_hint="ECCV 2020",
    ),
    "gwd": AlgorithmSpec(
        key="gwd",
        display_name="Gaussian Wasserstein Distance",
        family="gaussian_loss",
        upstream_config="gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc.py",
        angle_version="oc",
        paper_hint="ICML 2021",
    ),
    "kld": AlgorithmSpec(
        key="kld",
        display_name="Kullback-Leibler Divergence",
        family="gaussian_loss",
        upstream_config="kld/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc.py",
        angle_version="oc",
        paper_hint="NeurIPS 2021",
    ),
    "adaptive_gwd_kld": AlgorithmSpec(
        key="adaptive_gwd_kld",
        display_name="Adaptive GWD-KLD Hybrid",
        family="gaussian_loss",
        upstream_config="adaptive_gwd_kld_retinanet_r50_fpn_1x_dota_oc.py",
        angle_version="oc",
        paper_hint="Custom (aspect-ratio adaptive fusion of GWD and KLD)",
    ),
}


def get_algorithm(key: str) -> AlgorithmSpec:
    try:
        return MMROTATE_ALGORITHMS[key]
    except KeyError as exc:
        known = ", ".join(sorted(MMROTATE_ALGORITHMS))
        raise KeyError(f"Unknown algorithm '{key}'. Known algorithms: {known}") from exc


def list_algorithms() -> list[AlgorithmSpec]:
    return [MMROTATE_ALGORITHMS[key] for key in sorted(MMROTATE_ALGORITHMS)]
