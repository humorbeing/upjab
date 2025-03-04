import upjab_FirstPackage
from ShowOff_DiseaseAnomalyDetectionPipeline.video_to_crop import video_to_crop


from loguru import logger
log_name = "hahaha"
log_save = "{time}"
log_save = f"logs/{log_save}_{log_name}.log"
logger.add(log_save)

logger.info("Hello World")