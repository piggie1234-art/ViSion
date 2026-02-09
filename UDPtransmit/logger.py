import logging
import os
class InfoLogger():

    def __init__(self,name,path="./log/",level=logging.DEBUG) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.path = path
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(lineno)d : %(message)s')
        self.file_handler = logging.FileHandler(os.path.join(self.path,f"{name}.log"))
        self.file_handler.setLevel(level)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        
    def info(self,msg):
        self.logger.info(msg)
    def debug(self,msg):
        self.logger.debug(msg)
    def warning(self,msg):
        self.logger.warning(msg)
    def error(self,msg):
        self.logger.error(msg)
    def critical(self,msg):
        self.logger.critical(msg)
    def exception(self,msg):
        self.logger.exception(msg)
    def log(self,msg):
        self.logger.log(msg)