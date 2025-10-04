"""
Main entry point for CBIR system.
Executes pipelines according to the flags defined in general_config.py
"""

import logging
import logging.config

from config import general_config
from pipeline.descriptor_creator import precompute_descriptors
from pipeline.dev_pipeline import run_dev
from pipeline.test_pipeline import predict_and_save_results

logging.config.fileConfig("utils/logging.ini", disable_existing_loggers=False)
log = logging.getLogger(__name__)


if __name__ == "__main__":
    if general_config.PRECOMPUTE:
        log.info("Running precomputation pipeline...")
        precompute_descriptors()

    if general_config.DEV_PREDICTION:
        log.info("Running development evaluation pipeline...")
        run_dev()

    if general_config.TEST_PREDICTION:
        log.info("Running test/prediction pipeline...")
        predict_and_save_results()

    log.info("Finished.")
