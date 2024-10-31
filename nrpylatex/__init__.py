from nrpylatex.core.scanner import Scanner, ScannerError
from nrpylatex.core.parser import Parser, ParserError
from nrpylatex.core.generator import Generator, GeneratorError
from nrpylatex.utils.structures import IndexedSymbol, IndexedSymbolError
from nrpylatex.utils.exceptions import NRPyLaTeXError, NamespaceError
from nrpylatex.parse_latex import parse_latex

try:
    from nrpylatex.utils.ipython import ParseMagic

    def load_ipython_extension(ipython):
        ipython.register_magics(ParseMagic)
except ModuleNotFoundError:
    pass

__version__ = "1.4.0"
