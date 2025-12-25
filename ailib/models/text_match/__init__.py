# from .dense_baseline import DenseBaseline
from .dssm import Model as DSSM
from .multihead_attention_dssm import Model as MultiheadAttentionDSSM
# from .cdssm import CDSSM
# from .drmm import DRMM
# from .drmmtks import DRMMTKS
# from .esim import ESIM
# from .knrm import KNRM
# from .conv_knrm import ConvKNRM
# from .bimpm import BiMPM
# from .matchlstm import MatchLSTM
# from .arci import ArcI
# from .arcii import ArcII
# from .bert import Bert
# from .mvlstm import MVLSTM
# from .match_pyramid import MatchPyramid
# from .anmm import Model as aNMM
# from .hbmp import HBMP
# from .duet import DUET
# from .diin import DIIN
# from .match_srnn import MatchSRNN


def list_available() -> list:
    from ailib.models.base_model import BaseModel
    from ailib.tools.utils_inheritance import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseModel)
