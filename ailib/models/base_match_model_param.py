from ailib.param.param_table import ParamTable
from ailib.param import hyper_spaces
from ailib.param.param import Param

class BaseModelParam(ParamTable):
    """
    Notice that all parameters must be serialisable for the entire model
    to be serialisable. Therefore, it's strongly recommended to use python
    native data types to store parameters.
    """
    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__()
        self.add(Param(
            name='task',
            desc="Decides model output shape, loss, and metrics."
        ))
        self.add(Param(
            name='out_activation_func', value=None,
            desc="Activation function used in output layer."
        ))
        if with_embedding:
            self.add(Param(
                name='with_embedding', value=True,
                desc="A flag used help `auto` module. Shouldn't be changed."
            ))
            self.add(Param(
                name='embedding',
                desc='FloatTensor containing weights for the Embedding.',
                validator=lambda x: isinstance(x, np.ndarray)
            ))
            self.add(Param(
                name='embedding_input_dim',
                desc='Usually equals vocab size + 1. Should be set manually.'
            ))
            self.add(Param(
                name='embedding_output_dim',
                desc='Should be set manually.'
            ))
            self.add(Param(
                name='padding_idx', value=0,
                desc='If given, pads the output with the embedding vector at'
                        'padding_idx (initialized to zeros) whenever it encounters'
                        'the index.'
            ))
            self.add(Param(
                name='embedding_freeze', value=False,
                desc='`True` to freeze embedding layer training, '
                        '`False` to enable embedding parameters.'
            ))
        if with_multi_layer_perceptron:
            self.add(Param(
                name='with_multi_layer_perceptron', value=True,
                desc="A flag of whether a multiple layer perceptron is used. "
                        "Shouldn't be changed."
            ))
            self.add(Param(
                name='mlp_num_units', value=128,
                desc="Number of units in first `mlp_num_layers` layers.",
                hyper_space=hyper_spaces.quniform(8, 256, 8)
            ))
            self.add(Param(
                name='mlp_num_layers', value=3,
                desc="Number of layers of the multiple layer percetron.",
                hyper_space=hyper_spaces.quniform(1, 6)
            ))
            self.add(Param(
                name='mlp_num_fan_out', value=64,
                desc="Number of units of the layer that connects the multiple "
                        "layer percetron and the output.",
                hyper_space=hyper_spaces.quniform(4, 128, 4)
            ))
            self.add(Param(
                name='mlp_activation_func', value='relu',
                desc='Activation function used in the multiple '
                        'layer perceptron.'
            ))