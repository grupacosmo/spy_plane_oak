from Scripts.data_preprocessing import Preprocessing
from Scripts.gather_data import Data
from Scripts.model import Model


# TODO
#  1. Pack all into one class to reduce complexity
#  2. Test model
if __name__ == "__main__":
    data = Data()
    preprocessing = Preprocessing(data)
    preprocessing.size = (32,32)
    print(repr(preprocessing))
    preprocessing.normalize()
    preprocessing.resize()
    model = Model(preprocessing)
    model.build_cnn_model()
    model.fit_model()
    model.validate_model()
    model.save_model()
    ...
