from abc import ABCMeta, abstractmethod
from contracts import contract, ContractsMeta

class ExpectationInterface():
    __metaclass__ = ContractsMeta
    
    
    @abstractmethod
    @contract(value='array', dt='float,>=0', returns='None')
    def update(self, value, dt=1.0):
        pass
    

    @abstractmethod
    @contract(returns='array')
    def get_value(self):
        pass
