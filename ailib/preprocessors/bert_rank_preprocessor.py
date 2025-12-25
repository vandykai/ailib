"""Bert Preprocessor."""

from transformers import AutoTokenizer

from . import units
from ailib.data_pack import DataPack
from ailib.preprocessors.base_rank_preprocessor import BaseRankPreprocessor
from functools import partial

class BertRankPreprocessor(BaseRankPreprocessor):
    """
    Baisc preprocessor helper.

    :param mode: String, supported mode can be referred
        https://huggingface.co/transformers/model_doc/auto.html.

    """

    def __init__(self, mode: str = 'bert-base-uncased'):
        """Initialization."""
        super().__init__()
        self._tokenizer_inner = AutoTokenizer.from_pretrained(mode).encode
        self._tokenizer = lambda x:self._tokenizer_inner(x, add_special_tokens=False)

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """Tokenizer is all BertPreprocessor's need."""
        return

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()

        data_pack.apply_on_text(self._tokenizer,
                                mode='both', inplace=True, verbose=verbose)
        data_pack.append_text_length(inplace=True, verbose=verbose)
        data_pack.drop_empty(inplace=True)
        return data_pack
