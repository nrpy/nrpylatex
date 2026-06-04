import importlib

from .core.generator import Generator, GeneratorError
from .core.parser import Parser, ParserError
from .core.scanner import Scanner, ScannerError
from .parse_latex import parse_latex
from .utils.exceptions import NamespaceError, NRPyLaTeXError
from .utils.structures import IndexedSymbol, IndexedSymbolError

try:
    from IPython.core.interactiveshell import InteractiveShell

    from nrpylatex.utils.ipython import ParseMagic

    def load_ipython_extension(ipython: InteractiveShell) -> None:
        ipython.register_magics(ParseMagic)
except ModuleNotFoundError:
    pass

__version__ = importlib.metadata.version('nrpylatex')

__all__ = [
    'Generator',
    'GeneratorError',
    'Parser',
    'ParserError',
    'Scanner',
    'ScannerError',
    'parse_latex',
    'NamespaceError',
    'NRPyLaTeXError',
    'IndexedSymbol',
    'IndexedSymbolError',
]
