from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class TransfuserConfig:
    """Global TransFuser config."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    bkb_path: str = "/home/users/bencheng.liao/.cache/huggingface/hub/checkpoints/resnet34.a1_in1k/pytorch_model.bin"
    plan_anchor_path: str = "/home/users/bencheng.liao/PlanWrapper/playground/visualization/kmeans_navsim_traj_20.npy"

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    # 3.20 risk temporal attention flags
    use_risk_gate: bool = False
    use_historical_risk_attention: bool = False
    use_temporal_risk_cross_attention: bool = False
    use_memory_aux_loss: bool = False
    use_risk_aware_cls: bool = False
    use_step_brake_timing_loss: bool = False
    use_risk_shadow_evaluator: bool = False
    use_soft_risk_rescore: bool = False

    # LiDAR-based history risk token settings
    risk_history_num_frames: int = 4
    risk_history_dt: float = 0.5
    risk_token_dim: int = 12
    risk_front_x_min: float = 1.0
    risk_front_x_max: float = 32.0
    risk_front_y_abs: float = 1.8
    risk_lidar_min_z: float = 0.2
    risk_lidar_max_z: float = 3.0
    risk_lidar_min_points: int = 3
    risk_lidar_gap_percentile: float = 10.0
    risk_ego_front_offset: float = 2.0
    risk_ttc_max: float = 10.0
    risk_drac_max: float = 6.0
    risk_attention_layers: int = 1
    memory_aux_loss_weight: float = 0.2
    risk_gate_min_gap: float = 1.0
    risk_gate_cls_margin: float = 2.0

    # 3.20 stage-1 shadow evaluator. These settings add no trainable parameters.
    risk_shadow_agent_confidence: float = 0.35
    risk_shadow_gap_agreement: float = 3.0
    risk_shadow_min_reliability: float = 0.5
    risk_shadow_default_lead_width: float = 2.0
    risk_shadow_ego_width: float = 2.0
    risk_shadow_lateral_margin: float = 0.3
    risk_shadow_lead_accel_min: float = -4.0
    risk_shadow_lead_accel_max: float = 2.0
    risk_shadow_max_lead_speed: float = 40.0
    risk_shadow_clearance_cap: float = 40.0
    risk_shadow_warning_gap: float = 3.0
    risk_shadow_brake_accel: float = -0.5
    risk_shadow_brake_preparation_time: float = 1.0
    risk_shadow_jerk_free: float = 4.0
    risk_shadow_jerk_scale: float = 4.0
    risk_shadow_decel_free: float = 4.0
    risk_shadow_decel_scale: float = 4.0
    risk_shadow_clearance_weight: float = 0.45
    risk_shadow_timing_weight: float = 0.25
    risk_shadow_map_weight: float = 0.20
    risk_shadow_comfort_weight: float = 0.10
    risk_shadow_cls_margin: float = 1.0
    risk_shadow_logit_penalty: float = 1.0
    risk_shadow_counterfactual_topk: int = 3
    risk_shadow_thw_min: float = 0.5
    risk_shadow_thw_max: float = 2.5
    risk_shadow_longitudinal_lateral_tolerance: float = 0.75
    risk_shadow_longitudinal_heading_tolerance: float = 0.20
    risk_shadow_lateral_cost_advantage: float = 0.02

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    diff_loss_weight: float = 20.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    use_ema: bool = False
    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2


    # optmizer
    weight_decay: float = 1e-4
    lr_steps = [70]
    optimizer_type = "AdamW"
    scheduler_type = "MultiStepLR"
    cfg_lr_mult = 0.5
    opt_paramwise_cfg = {
        "name":{
            "image_encoder":{
                "lr_mult": cfg_lr_mult
            }
        }
    }
    # optimizer=dict(
    #     type="AdamW",
    #     lr=1e-4,
    #     weight_decay=1e-6,
    # )
    # scheduler=dict(
    #     type="MultiStepLR",
    #     milestones=[90],
    #     gamma=0.1,
    # )

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
