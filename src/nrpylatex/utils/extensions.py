from __future__ import annotations

import re
from typing import Any

from IPython.core.magic import Magics, line_cell_magic, magics_class
from sympy import Function

from ..parse_latex import ParsedNamespace, parse_latex
from .exceptions import NRPyLaTeXError
from .structures import IndexedSymbol


class IPythonNamespace(ParsedNamespace):
    def __init__(self, namespace: ParsedNamespace, sentence: str) -> None:
        super().__init__(namespace._extracted_vars, namespace._overridden_keys)
        self.sentence = sentence

    def _repr_latex_(self) -> str:
        lines = (line for line in self.sentence.splitlines() if not line.lstrip().startswith('%'))
        return '$$' + '\n'.join(lines).strip() + '$$'


@magics_class
class ParseMagic(Magics):
    """NRPyLaTeX IPython Magic"""

    @line_cell_magic
    def parse_latex(self, line: str, cell: str | None = None) -> Any:
        match = re.match(r'\s*--([^\s]+)\s*', line)

        kwargs: list[str] = []
        while match:
            kwargs.append(match.group(1))
            line = line[match.span()[-1] :]
            match = re.match(r'\s*--([^\s]+)\s*', line)
        reset, debug = 'reset' in kwargs, 'debug' in kwargs

        try:
            sentence = line if cell is None else cell
            parsed_result = parse_latex(sentence, reset=reset, debug=debug)
            if not isinstance(parsed_result, ParsedNamespace):
                return parsed_result

            if self.shell is not None:
                for key in parsed_result:
                    value = getattr(parsed_result, key)
                    if isinstance(value, IndexedSymbol):
                        self.shell.user_ns[key] = value.structure
                    elif isinstance(value, Function('Constant')):
                        self.shell.user_ns[key] = value.args[0]
                    else:
                        self.shell.user_ns[key] = value

            return IPythonNamespace(parsed_result, sentence)

        except NRPyLaTeXError as e:
            print(f'{type(e).__name__}: {e}')
