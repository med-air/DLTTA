from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       - This class can be used in both server and client side
       - This class is an operator which does not cache any states inside.
    """
    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    
    # @abstractmethod
    # def validate(self, val_data, device, args=None):
    #     pass

    

    # @abstractmethod
    # def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
    #     pass
