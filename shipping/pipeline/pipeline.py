import os
import sys
import uuid
from collections import namedtuple
from datetime import datetime
from threading import Thread

import pandas as pd

from shipping.component.data_ingestion import DataIngestion
from shipping.config.configuration import Configuartion
from shipping.constant import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME
from shipping.entity.artifact_entity import (DataIngestionArtifact,
                                             DataTransformationArtifact,
                                             DataValidationArtifact,
                                             ModelEvaluationArtifact,
                                             ModelPusherArtifact,
                                             ModelTrainerArtifact)
from shipping.exception import ShippingException
from shipping.logger import get_log_file_name, logging

Experiment = namedtuple(
    "Experiment",
    [
        "experiment_id",
        "initialization_timestamp",
        "artifact_time_stamp",
        "running_status",
        "start_time",
        "stop_time",
        "execution_time",
        "message",
        "experiment_file_path",
        "accuracy",
        "is_model_accepted",
    ],
)


class Pipeline(Thread):
    experiment: Experiment = Experiment(*([None] * 11))
    experiment_file_path = None

    def __init__(self, config: Configuartion) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            Pipeline.experiment_file_path = os.path.join(
                config.training_pipeline_config.artifact_dir,
                EXPERIMENT_DIR_NAME,
                EXPERIMENT_FILE_NAME,
            )
            super().__init__(daemon=False, name="pipeline")
            self.config = config
        except Exception as e:
            raise ShippingException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.config.get_data_ingestion_config()
            )
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ShippingException(e, sys) from e

   
    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")

                return Pipeline.experiment

            logging.info("Pipeline starting.")

            experiment_id = str(uuid.uuid4())

            Pipeline.experiment = Experiment(
                experiment_id=experiment_id,
                initialization_timestamp=self.config.time_stamp,
                artifact_time_stamp=self.config.time_stamp,
                running_status=True,
                start_time=datetime.now(),
                stop_time=None,
                execution_time=None,
                experiment_file_path=Pipeline.experiment_file_path,
                is_model_accepted=None,
                message="Pipeline has been started.",
                accuracy=None,
            )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")

            self.save_experiment()

            data_ingestion_artifact = self.start_data_ingestion()

            
        except Exception as e:
            raise ShippingException(e, sys) from e

    def run(self):
        try:
            self.run_pipeline()
        
        except Exception as e:
            raise e

    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                
                experiment_dict = experiment._asdict()
                
                experiment_dict: dict = {
                    key: [value] for key, value in experiment_dict.items()
                }

                experiment_dict.update(
                    {
                        "created_time_stamp": [datetime.now()],
                        "experiment_file_path": [
                            os.path.basename(Pipeline.experiment.experiment_file_path)
                        ],
                    }
                )

                experiment_report = pd.DataFrame(experiment_dict)

                os.makedirs(
                    os.path.dirname(Pipeline.experiment_file_path), exist_ok=True
                )
                
                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(
                        Pipeline.experiment_file_path,
                        index=False,
                        header=False,
                        mode="a",
                    )
                else:
                    experiment_report.to_csv(
                        Pipeline.experiment_file_path,
                        mode="w",
                        index=False,
                        header=True,
                    )
            else:
                print("First start experiment")
        
        except Exception as e:
            raise ShippingException(e, sys) from e

    @classmethod
    def get_experiments_status(cls, limit: int = 5) -> pd.DataFrame:
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
               
                limit = -1 * int(limit)
                
                return df[limit:].drop(
                    columns=["experiment_file_path", "initialization_timestamp"], axis=1
                )
            
            else:
                return pd.DataFrame()
        
        except Exception as e:
            raise ShippingException(e, sys) from e
