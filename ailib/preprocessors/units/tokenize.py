import nltk
import jieba_fast as jieba

from .unit import Unit

class NLTKTokenize(Unit):
    """Process unit for text tokenization."""

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        return nltk.word_tokenize(input_)

class JiebaTokenize(Unit):
    """Process unit for text tokenization."""

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        return jieba.lcut(input_)