import unicodedata
from .unit import Unit

class TextNormalize(Unit):
    """Process unit to remove digits."""
    def __init__(self, form='NFKC'):
        """
        :param from: Valid values for form are 'NFC', 'NFKC', 'NFD', and 'NFKD'..
        """
        self._form = form

    def transform(self, input_: str) -> str:
        """
        Remove digits from list of tokens.

        :param input_: raw textual input.

        :return str: html cleand text.
        """
        text = unicodedata.normalize(self._form, input_)
        return text