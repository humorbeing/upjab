from loguru import logger
log_name = 'test_loguru'
log_path = 'logs/{time}_' + f'{log_name}.log'

from upjab import AP
log_path = AP(log_path)

from upjab.tool import remove_folder
remove_folder(AP('logs'))

handler_id = logger.add(log_path)
num = 5
logger.info(f'experiment setup: num_inducing_points {num}')

logger.remove(handler_id)