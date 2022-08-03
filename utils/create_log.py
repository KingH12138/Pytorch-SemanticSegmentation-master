import logging


def logger_get(log_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    file_logger = logging.getLogger('FileLogger')

    # 向文件输出日志信息
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_logger.addHandler(file_handler)
    return file_logger


