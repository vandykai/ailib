"""Exercise Preprocessor."""

from .base_preprocessor import BasePreprocessor
from .utils import chain_transform

from ailib import preprocessors


class ExercisePreprocessor(BasePreprocessor):
    """
    Exercise preprocessor helper.

    Example:
        >>> train_data = ["example one", "example two"]
        >>> preprocessor = preprocessors.BasicPreprocessor()
        >>> processed_train_data = list(map(preprocessor.transform, train_data))

    """

    def __init__(self):
        """Initialization."""
        super().__init__()
        self.context['latex_tokens'] = list(preprocessors.units.latex_convert.LatexConvert.latex_word_map.values())

    @classmethod
    def _default_units(cls) -> list:
        """Prepare needed process units."""
        return [
            preprocessors.units.html_removal.HTMLRemoval(),
            preprocessors.units.latex_convert.LatexConvert(),
            preprocessors.units.text_normalize.TextNormalize(form='NFKC')
        ]

    def transform(self, data: str, verbose: int = 1) -> str:
        """
        Apply transformation on data.

        :param data: Input str to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`str` object.
        """
        data = chain_transform(self._default_units())(data)
        return data
        
