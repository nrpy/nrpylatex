import sys

if sys.version_info >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata


from nrpylatex.core.generator import Generator, GeneratorError
from nrpylatex.core.parser import Parser, ParserError
from nrpylatex.core.scanner import Scanner, ScannerError
from nrpylatex.parse_latex import parse_latex
from nrpylatex.utils.exceptions import NamespaceError, NRPyLaTeXError
from nrpylatex.utils.structures import IndexedSymbol, IndexedSymbolError

try:
    from IPython.core.interactiveshell import InteractiveShell

    from nrpylatex.utils.extensions import ParseMagic

    def load_ipython_extension(ipython: InteractiveShell) -> None:
        ipython.register_magics(ParseMagic)
except ModuleNotFoundError:
    pass

__version__ = importlib_metadata.version('nrpylatex')

__all__ = [
    'Generator',
    'GeneratorError',
    'IndexedSymbol',
    'IndexedSymbolError',
    'NRPyLaTeXError',
    'NamespaceError',
    'Parser',
    'ParserError',
    'Scanner',
    'ScannerError',
    'parse_latex',
]
