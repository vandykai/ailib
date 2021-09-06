import re
from bs4 import BeautifulSoup
from .unit import Unit

class HTMLRemoval(Unit):
    """Process unit to remove html tag."""
    nbsp_blank_reg = re.compile(r"<u>((&nbsp;){1,})<\/u>")
    def replace_nbsp_blank(self, input_: str)-> str:
        """
        replace special nbsp tag，eg:<u>&nbsp;</u>->_
        """
        ret = ''
        pos = 0
        for tag in self.nbsp_blank_reg.finditer(input_):
            st, ed = tag.span()
            if st > pos:
                ret += input_[pos:st]
            group1 = tag.group(1)
            if group1 is not None:
                ret += "_" * group1.count("&nbsp;")
            pos = ed
        ret += input_[pos:]
        return ret

    def transform(self, input_: str) -> str:
        """
        remove html tag.

        :param input_: raw textual input.

        :return str: html cleand text.
        """
        text = self.replace_nbsp_blank(input_)
        soup = BeautifulSoup(text, "html.parser")
        info = [s.extract() for s in soup('table')]
        info = [s.extract() for s in soup('img')]
        info = [s.extract() for s in soup('video')]
        info = [s.extract() for s in soup('audio')]
        text = soup.get_text()
        return text