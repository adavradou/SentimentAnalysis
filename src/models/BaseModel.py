from abc import ABC, abstractmethod
#from utils.Config import Config
import sys
sys.path.insert(0, '../utils')
from Config import *

class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

    @abstractmethod
    def create_data_pipeline(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass