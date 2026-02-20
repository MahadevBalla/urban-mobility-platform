"""
Main Travel Demand Estimation Pipeline.

End-to-end pipeline for estimating travel demand from telecom data.
Orchestrates data loading, preprocessing, stay detection, trip generation,
and OD matrix creation.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from src.data_fusion import MultiSourceFusion
from src.data_ingestion import CellTowerLoader, TelecomDataLoader, ZoneLoader
from src.od_matrix import ODMatrixGenerator
from src.preprocessing import TelecomPreprocessor, UserFilter
from src.stay_detection import HomeWorkInference, StayPointDetector
from src.trip_generation import TripExpander, TripGenerator
from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TravelDemandPipeline:
    """
    End-to-end travel demand estimation pipeline.

    Orchestrates the full workflow from raw telecom data to OD matrices:
    1. Data Loading - Load CDR, XDR, network data
    2. Preprocessing - Clean, filter, standardize
    3. Cell Location Inference - Build cell tower location database
    4. Stay Detection - Extract stay points from trajectories
    5. Home/Work Inference - Classify stay points
    6. Trip Generation - Extract trips between stay points
    7. Trip Expansion - Scale to population level
    8. OD Matrix Creation - Aggregate trips to zone-level OD flows

    Processing modes (config.general.processing.mode):
        - "full" - All DataFrames held in memory (original behaviour, dev/sample only)
        "chunked" - preprocessed written to disk after zone creation; steps 5-7
                    load users in batches to reduce memory usage (recommended for large datasets)

    Example:
        >>> pipeline = TravelDemandPipeline()
        >>> results = pipeline.run("data/raw")
        >>> pipeline.save_results(results, "data/outputs")
    """

    def __init__(self, config: Optional[Union[str, Path, Config]] = None):
        """
        Initialize pipeline.

        Args:
            config: Configuration object, path to config file, or None for defaults.
        """
        if isinstance(config, (str, Path)):
            self.config = Config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = get_config()

        # Initialize components
        self.data_loader = TelecomDataLoader(self.config)
        self.preprocessor = TelecomPreprocessor(self.config)
        self.user_filter = UserFilter(self.config)
        self.cell_loader = CellTowerLoader(self.config)
        self.zone_loader = ZoneLoader(self.config)
        self.stay_detector = StayPointDetector(self.config)
        self.home_work_inference = HomeWorkInference(self.config)
        self.trip_generator = TripGenerator(self.config)
        self.trip_expander = TripExpander(self.config)
        self.od_generator = ODMatrixGenerator(self.zone_loader, self.config)

        # Store intermediate results
        self._results: Dict = {}

        # Processing mode from config
        proc = self.config.get("general.processing", {})
        self._mode = proc.get("mode", "full")
        self._chunk_size = proc.get("chunk_size_users", 1000)
        self._intermediate_fmt = proc.get("intermediate_format", "parquet")
        self._intermediate_dir = Path(proc.get("intermediate_dir", "data/intermediate"))

    def run(
        self,
        data_path: Union[str, Path],
        sample_fraction: Optional[float] = None,
        steps: Optional[list] = None,
    ) -> Dict:
        """
        Run the full pipeline.

        Args:
            data_path: Path to directory containing input data files.
            sample_fraction: Optional fraction of data to sample (for testing).
            steps: Optional list of steps to run. Default runs all steps.

        Returns:
            Dictionary containing all intermediate and final results.
        """
        data_path = Path(data_path)

        all_steps = [
            "load_data",
            "fuse_data",
            "preprocess",
            "infer_cell_locations",
            "create_zones",
            "detect_stays",
            "infer_home_work",
            "generate_trips",
            "expand_trips",
            "generate_od_matrix",
        ]

        steps = steps or all_steps

        logger.info("Starting Travel Demand Estimation Pipeline")

        if self._mode == "chunked":
            self._intermediate_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Processing mode: chunked "
                f"(chunk_size={self._chunk_size} users, "
                f"format={self._intermediate_fmt})"
            )
        else:
            logger.info("Processing mode: full (in-memory)")

        # Step 1: Load Data
        if "load_data" in steps:
            logger.info("[Step 1/10] Loading data...")
            self._results["raw_data"] = self.data_loader.load_all(
                data_path, sample_fraction
            )

        # Step 2: Multi-source fusion
        if "fuse_data" in steps:
            logger.info("[Step 2/10] Fusing multi-source telecom data...")
            raw = self._results["raw_data"]
            fusion = MultiSourceFusion(self.config)
            fused = fusion.fuse(
                cdr_df=raw.get("cdr"),
                xdr_df=raw.get("xdr"),
                network_4g_df=raw.get("network_4g"),
                network_5g_df=raw.get("network_5g"),
            )
            self._results["fused_data"] = fused
            logger.info(f"Fusion summary: {fusion.get_fusion_summary(fused)}")

        # Step 3: Preprocess
        if "preprocess" in steps:
            logger.info("[Step 3/10] Preprocessing data...")
            # Use fused_data if available, fall back to raw XDR+CDR
            fused = self._results.get("fused_data")
            raw = self._results.get("raw_data", {})
            preprocessed = self.preprocessor.process(
                cdr_df=raw.get("cdr"),
                xdr_df=fused if fused is not None else raw.get("xdr"),
            )
            preprocessed = self.preprocessor.add_derived_features(preprocessed)

            # Get user stats before filtering
            self._results["user_stats"] = self.preprocessor.get_user_summary(
                preprocessed
            )

            # Filter users
            valid_users = self.user_filter.filter_users(preprocessed)
            preprocessed = preprocessed[
                preprocessed["imsi"].isin(valid_users)
            ].reset_index(drop=True)
            self._results["filter_stats"] = self.user_filter.filter_stats

            if self._mode == "chunked":
                # Write to disk; do NOT keep in self._results
                self._write_intermediate(preprocessed, "preprocessed")
                # Keep only the list of valid users in memory
                self._results["valid_users"] = preprocessed["imsi"].unique().tolist()
                del preprocessed
            else:
                self._results["preprocessed"] = preprocessed

        # Step 4: Infer Cell Locations
        if "infer_cell_locations" in steps:
            logger.info("[Step 4/10] Inferring cell tower locations...")
            min_samples = int(self.config.get("cell_towers.min_hull_samples", 5))
            preprocessed = self._get_preprocessed()
            if preprocessed is not None:
                self.cell_loader.infer_from_xdr(preprocessed, min_samples=min_samples)
                self.cell_loader.infer_tac_locations(preprocessed)

                # Add locations to data if missing
                updated = self.cell_loader.add_locations_to_df(preprocessed)

                if self._mode == "chunked":
                    # Overwrite intermediate with location-enriched version
                    self._write_intermediate(updated, "preprocessed")
                    del preprocessed, updated
                else:
                    self._results["preprocessed"] = updated

        # Step 5: Create Zones
        if "create_zones" in steps:
            logger.info("[Step 5/10] Creating zone definitions...")
            preprocessed = self._get_preprocessed()
            if preprocessed is not None:
                self.zone_loader.create_tac_zones(preprocessed)
                # preprocessed no longer needed after zone creation in chunked mode
                if self._mode == "chunked":
                    del preprocessed

        # Step 6: Detect Stay Points
        if "detect_stays" in steps:
            logger.info("[Step 6/10] Detecting stay points...")
            self._results["stay_points"] = self._run_stay_detection()

        # Step 7: Infer Home/Work
        if "infer_home_work" in steps:
            logger.info("[Step 7/10] Inferring home and work locations...")
            self._results["stay_points"] = self.home_work_inference.infer(
                self._results["stay_points"],
                # In chunked mode preprocessed is on disk; home/work inference
                # only uses stay_points directly so pass None
                self._results.get("preprocessed"),
            )
            self._results["home_work_summary"] = (
                self.home_work_inference.get_home_work_summary(
                    self._results["stay_points"]
                )
            )

        # Step 8: Generate Trips
        if "generate_trips" in steps:
            logger.info("[Step 8/10] Generating trips...")
            self._results["trips"] = self._run_trip_generation()
            self._results["trip_table"] = self.trip_generator.get_trip_table(
                self._results["trips"]
            )

        # Step 9: Expand Trips
        if "expand_trips" in steps:
            logger.info("[Step 9/10] Expanding trips to population level...")

            # Build home zone mapping
            home_zones = self._build_home_zones()

            # Get zone populations if available
            zone_pops = self._build_zone_populations()

            self._results["trips"] = self.trip_expander.expand(
                self._results["trips"],
                self._results.get("user_stats", pd.DataFrame()),
                zone_pops,
                home_zones,
            )
            self._results["expansion_summary"] = (
                self.trip_expander.get_expansion_summary(self._results["trips"])
            )

        # Step 10: Generate OD Matrix
        if "generate_od_matrix" in steps:
            logger.info("[Step 10/10] Generating OD matrix...")

            # Full OD matrix
            self._results["od_matrix"] = self.od_generator.generate(
                self._results["trips"]
            )

            # By purpose
            self._results["od_by_purpose"] = self.od_generator.generate_by_purpose(
                self._results["trips"]
            )

            # By time period
            self._results["od_by_time"] = self.od_generator.generate_by_time_period(
                self._results["trips"]
            )

            # Summary
            self._results["od_summary"] = self.od_generator.get_summary_statistics(
                self._results["od_matrix"]
            )

        logger.info("Pipeline Complete")
        self._log_summary()

        return self._results

    # Chunked execution helpers
    def _run_stay_detection(self) -> pd.DataFrame:
        """
        Run stay detection, chunked in chunked mode.

        In full mode: passes full preprocessed DF to StayPointDetector.detect()
        (original behaviour, unchanged).

        In chunked mode: loads users in batches of chunk_size_users from the
        intermediate parquet file, calls detect() per batch, concatenates
        stay_points. Peak RAM per batch =
            chunk_size_users * avg_records_per_user * record_size
        instead of full dataset * record_size.
        """
        if self._mode != "chunked":
            return self.stay_detector.detect(self._results["preprocessed"])

        users = self._results["valid_users"]
        all_stays = []

        for _, batch_df in self._iter_user_chunks(users):
            batch_stays = self.stay_detector.detect(batch_df)
            if len(batch_stays) > 0:
                all_stays.append(batch_stays)
            del batch_df

        if all_stays:
            return pd.concat(all_stays, ignore_index=True)

        return pd.DataFrame(
            columns=[
                "user_id",
                "stay_id",
                "latitude",
                "longitude",
                "cell_id",
                "tac",
                "first_seen",
                "last_seen",
                "observation_count",
                "visit_count",
                "total_duration",
                "location_type",
            ]
        )

    def _run_trip_generation(self) -> pd.DataFrame:
        """
        Run trip generation, chunked in chunked mode.

        TripGenerator.generate() needs both stay_points and observations
        (preprocessed) per user. In chunked mode observations are loaded
        per batch from disk; stay_points are small and already in memory.
        """
        if self._mode != "chunked":
            return self.trip_generator.generate(
                self._results["stay_points"],
                self._results.get("preprocessed"),
            )

        users = self._results["valid_users"]
        stay_points = self._results["stay_points"]
        all_trips = []

        for batch_users, batch_obs in self._iter_user_chunks(users):
            batch_stays = stay_points[stay_points["user_id"].isin(batch_users)]
            batch_trips = self.trip_generator.generate(batch_stays, batch_obs)
            if len(batch_trips) > 0:
                all_trips.append(batch_trips)
            del batch_obs, batch_stays

        if all_trips:
            return pd.concat(all_trips, ignore_index=True)
        return pd.DataFrame()

    def _iter_user_chunks(self, users: list):
        """
        Yield (batch_user_list, batch_df) by loading user chunks from the
        intermediate preprocessed file.

        Yields:
            (batch_users: List[str], batch_df: pd.DataFrame)
        """
        path = self._intermediate_path("preprocessed")

        for i in range(0, len(users), self._chunk_size):
            batch_users = users[i : i + self._chunk_size]

            if self._intermediate_fmt == "parquet":
                # pandas parquet filter pushdown - reads only matching row groups
                batch_df = pd.read_parquet(
                    path,
                    filters=[("imsi", "in", batch_users)],
                )
            else:
                # CSV fallback: must read full file and filter in-memory
                # Acceptable only for sample/dev; chunked mode should use parquet
                full = pd.read_csv(path, parse_dates=["timestamp"])
                batch_df = full[full["imsi"].isin(batch_users)]
                del full

            logger.debug(
                f"Loaded chunk {i // self._chunk_size + 1}: "
                f"{len(batch_users)} users, {len(batch_df)} records"
            )
            yield batch_users, batch_df

    # Intermediate file I/O
    def _intermediate_path(self, name: str) -> Path:
        ext = "parquet" if self._intermediate_fmt == "parquet" else "csv"
        return self._intermediate_dir / f"{name}.{ext}"

    def _write_intermediate(self, df: pd.DataFrame, name: str) -> None:
        path = self._intermediate_path(name)
        if self._intermediate_fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.debug(f"Wrote intermediate '{name}' to {path} ({len(df)} records)")

    def _get_preprocessed(self) -> Optional[pd.DataFrame]:
        """
        Return preprocessed DataFrame.

        In full mode: returns from self._results.
        In chunked mode: loads from disk (needed for steps 3-4 which
        require the full dataset to infer cell locations and zones).
        """
        if self._mode != "chunked":
            return self._results.get("preprocessed")

        path = self._intermediate_path("preprocessed")
        if not path.exists():
            logger.warning(f"Intermediate file not found: {path}")
            return None

        if self._intermediate_fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, parse_dates=["timestamp"])

    # Small helpers (unchanged logic, extracted for clarity)
    def _build_home_zones(self) -> Dict[str, str]:
        home_zones = {}
        if "stay_points" in self._results:
            home_stays = self._results["stay_points"][
                self._results["stay_points"]["location_type"] == "home"
            ]
            for _, stay in home_stays.iterrows():
                tac = stay.get("tac") or stay.get("cell_id")
                if tac:
                    home_zones[stay["user_id"]] = str(tac)
        return home_zones

    def _build_zone_populations(self) -> Optional[Dict[str, int]]:
        if self.zone_loader.zone_count == 0:
            return None
        zones_df = self.zone_loader.to_dataframe()
        return dict(
            zip(
                zones_df["zone_id"],
                zones_df["population"].fillna(10000),
            )
        )

    def _log_summary(self) -> None:
        """Log summary of pipeline results."""
        r = self._results

        logger.info("-" * 30 + "Pipeline Summary" + "-" * 30)

        if "filter_stats" in r:
            logger.info(f"Users: {r['filter_stats'].get('valid_users', 'N/A')}")

        if "stay_points" in r:
            logger.info(f"Stay points: {len(r['stay_points'])}")
            home_count = (r["stay_points"]["location_type"] == "home").sum()
            work_count = (r["stay_points"]["location_type"] == "work").sum()
            logger.info(f"  Homes: {home_count}, Work locations: {work_count}")

        if "trips" in r:
            logger.info(f"Trips: {len(r['trips'])}")
            if "expanded_trips" in r["trips"].columns:
                logger.info(
                    f"  Expanded total: {r['trips']['expanded_trips'].sum():.0f}"
                )

        if "od_summary" in r:
            logger.info(f"OD pairs: {r['od_summary']['non_zero_pairs']}")
            logger.info(f"Total flow: {r['od_summary']['total_flow']:.0f}")

    def save_results(
        self, results: Optional[Dict] = None, output_dir: Union[str, Path] = "outputs"
    ) -> None:
        """
        Save pipeline results to files.

        Args:
            results: Results dictionary. Uses internal results if None.
            output_dir: Output directory path.
        """
        results = results or self._results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to {output_dir}")

        # Save stay points
        if "stay_points" in results:
            results["stay_points"].to_csv(output_dir / "stay_points.csv", index=False)

        # Save trips
        if "trips" in results:
            results["trips"].to_csv(output_dir / "trips.csv", index=False)

        # Save trip table
        if "trip_table" in results:
            results["trip_table"].to_csv(output_dir / "trip_table.csv", index=False)

        # Save OD matrix
        if "od_matrix" in results:
            self.od_generator.to_csv(results["od_matrix"], output_dir / "od_matrix.csv")

        # Save OD by purpose
        if "od_by_purpose" in results:
            for purpose, matrix in results["od_by_purpose"].items():
                matrix.to_csv(output_dir / f"od_matrix_{purpose}.csv", index=False)

        # Save home/work summary
        if "home_work_summary" in results:
            results["home_work_summary"].to_csv(
                output_dir / "home_work_summary.csv", index=False
            )

        # Save zones
        if self.zone_loader.zone_count > 0:
            self.zone_loader.save(output_dir / "zones.csv")

        # Save cell locations
        if self.cell_loader.cell_count > 0:
            self.cell_loader.save(output_dir / "cell_locations.csv")

        logger.info(f"Results saved to {output_dir}")

    @property
    def results(self) -> Dict:
        """Get pipeline results."""
        return self._results


def run_pipeline(
    data_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    sample_fraction: Optional[float] = None,
) -> Dict:
    """
    Convenience function to run the full pipeline.

    Args:
        data_path: Path to input data directory.
        config_path: Optional path to configuration file.
        output_dir: Optional output directory (saves results if provided).
        sample_fraction: Optional fraction to sample for testing.

    Returns:
        Dictionary of pipeline results.
    """
    pipeline = TravelDemandPipeline(config_path)
    results = pipeline.run(data_path, sample_fraction)

    if output_dir:
        pipeline.save_results(results, output_dir)

    return results
