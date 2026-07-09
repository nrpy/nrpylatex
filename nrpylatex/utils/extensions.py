import re
from typing import Any, List, Optional

from IPython.core.magic import Magics, line_cell_magic, magics_class

from ..parse_latex import ParsedNamespace, parse_latex
from .exceptions import NRPyLaTeXError


class IPythonNamespace(ParsedNamespace):
    def __init__(self, parsed_ns: ParsedNamespace, sentence: str) -> None:
        super().__init__(parsed_ns._variables, parsed_ns._overridden)
        self.sentence = sentence

    def _repr_latex_(self) -> str:
        return rf'\[{self.sentence}\]'


@magics_class
class ParseMagic(Magics):
    """NRPyLaTeX IPython Magic"""

    @line_cell_magic
    def parse_latex(self, line: str, cell: Optional[str] = None) -> Any:
        match = re.match(r'\s*--([^\s]+)\s*', line)

        kwargs: List[str] = []
        while match:
            kwargs.append(match.group(1))
            line = line[match.span()[-1] :]
            match = re.match(r'\s*--([^\s]+)\s*', line)
        reset, debug = 'reset' in kwargs, 'debug' in kwargs

        try:
            sentence = line if cell is None else cell
            result = parse_latex(sentence, reset=reset, debug=debug)
            if not isinstance(result, ParsedNamespace):
                return result
            return IPythonNamespace(result, sentence)

        except NRPyLaTeXError as e:
            print(f'{type(e).__name__}: {e}')
