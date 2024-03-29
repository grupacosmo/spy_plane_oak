from Scripts.data_preprocessing import Preprocessing
from Scripts.gather_data import Data
# from Scripts.model import Model


if __name__ == "__main__":
    data = Data()
    preprocessing = Preprocessing(data)
    print(repr(preprocessing))
    ...
