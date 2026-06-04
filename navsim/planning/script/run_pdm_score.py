from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _reset_agent_temporal_context(agent: AbstractAgent) -> None:
    """Reset optional temporal state before processing an unrelated log."""
    if hasattr(agent, "reset_temporal_context"):
        agent.reset_temporal_context()


def _get_agent_temporal_flags(agent: AbstractAgent) -> Dict[str, float]:
    """Read optional temporal debug flags for CSV output."""
    if hasattr(agent, "get_temporal_debug_info"):
        debug_info = agent.get_temporal_debug_info()
        return {
            "temporal_reference_active": float(bool(debug_info.get("temporal_reference_active", False))),
            "previous_ego_delta_active": float(bool(debug_info.get("previous_ego_delta_active", False))),
        }

    return {
        "temporal_reference_active": float(bool(getattr(agent, "_last_temporal_reference_active", False))),
        "previous_ego_delta_active": float(bool(getattr(agent, "_last_previous_ego_delta_active", False))),
    }


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    if len(args) == 0:
        return []

    log_names = [a["log_file"] for a in args]
    requested_tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    logger.info("PDM agent target: %s", cfg.agent.get("_target_", None))
    logger.info("PDM checkpoint_path: %s", cfg.agent.get("checkpoint_path", None))

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    metric_tokens = set(metric_cache_loader.tokens)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = requested_tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    scene_tokens = set(scene_loader.tokens)

    pdm_results: List[Dict[str, Any]] = []
    total_tokens_to_evaluate = sum(
        1 for a in args for token in a["tokens"] if token in scene_tokens and token in metric_tokens
    )
    processed_tokens = 0

    for log_item in args:
        log_file = log_item["log_file"]
        tokens_to_evaluate = [
            token for token in log_item["tokens"] if token in scene_tokens and token in metric_tokens
        ]
        _reset_agent_temporal_context(agent)
        logger.info(
            f"Processing log {log_file} with {len(tokens_to_evaluate)} scenarios "
            f"in thread_id={thread_id}, node_id={node_id}"
        )

        for token in tokens_to_evaluate:
            processed_tokens += 1
            logger.info(
                f"Processing scenario {processed_tokens} / {total_tokens_to_evaluate} "
                f"in thread_id={thread_id}, node_id={node_id}"
            )
            score_row: Dict[str, Any] = {
                "token": token,
                "valid": True,
                "temporal_reference_active": 0.0,
                "previous_ego_delta_active": 0.0,
            }
            try:
                metric_cache_path = metric_cache_loader.metric_cache_paths[token]
                with lzma.open(metric_cache_path, "rb") as f:
                    metric_cache: MetricCache = pickle.load(f)

                agent_input = scene_loader.get_agent_input_from_token(token)
                if agent.requires_scene:
                    scene = scene_loader.get_scene_from_token(token)
                    trajectory = agent.compute_trajectory(agent_input, scene)
                else:
                    trajectory = agent.compute_trajectory(agent_input)
                score_row.update(_get_agent_temporal_flags(agent))

                pdm_result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=trajectory,
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                )
                score_row.update(asdict(pdm_result))
            except Exception as e:
                logger.warning(f"----------- Agent failed for token {token}:")
                traceback.print_exc()
                score_row["valid"] = False

            pdm_results.append(score_row)
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    metric_tokens = set(metric_cache_loader.tokens)
    scene_tokens = set(scene_loader.tokens)
    tokens_to_evaluate = [token for token in scene_loader.tokens if token in metric_tokens]
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - scene_tokens)
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    score_rows: List[Dict[str, Any]] = worker_map(worker, run_pdm_score, data_points)
    if len(score_rows) > 0 and isinstance(score_rows[0], list):
        score_rows = [row for worker_rows in score_rows for row in worker_rows]

    pdm_score_df = pd.DataFrame(score_rows)
    if len(pdm_score_df) == 0:
        logger.warning("No scenarios were evaluated. Empty result file will be saved.")
        save_path = Path(cfg.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        pdm_score_df.to_csv(save_path / f"{timestamp}.csv", index=False)
        return

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    numeric_columns = pdm_score_df.drop(columns=["token", "valid"], errors="ignore").select_dtypes(include="number").columns
    average_row = pdm_score_df[numeric_columns].mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    result_path = save_path / f"{timestamp}.csv"
    pdm_score_df.to_csv(result_path, index=False)
    valid_score_mean = pdm_score_df[pdm_score_df["token"] != "average"]["score"].mean(skipna=True)

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {valid_score_mean}.
            Results are stored in: {result_path}.
        """
    )


if __name__ == "__main__":
    main()
