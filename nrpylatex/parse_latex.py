""" NRPyLaTeX: LaTeX Interface to SymPy (CAS) for General Relativity """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

from nrpylatex.core.parser import Parser
from nrpylatex.utils.exceptions import NRPyLaTeXError, NamespaceError
from nrpylatex.utils.structures import IndexedSymbol
from sympy import Function, Symbol
from inspect import currentframe
import re

def parse_latex(sentence, reset=False, debug=False, namespace=None):
    """ Convert LaTeX to SymPy

        :arg: latex sentence (str)
        :arg: reset namespace (bool)
        :arg: debug parse_latex (bool)
        :arg: import namespace (dict)
        :return: namespace or expression
    """
    if reset: Parser.initialize(reset=True)
    if namespace:
        for symbol in namespace:
            function = Function('Tensor')(Symbol(symbol, real=True))
            structure = namespace[symbol]
            if not isinstance(structure, list):
                raise NamespaceError('cannot import variable of type %s, only list' % type(structure))
            dimension = len(structure)
            i = 0
            while isinstance(structure[i], list):
                if len(structure[i] != dimension):
                    raise NamespaceError('inconsistent dimension in \'%s\'' % symbol)
                i += 1
            Parser._namespace[symbol] = IndexedSymbol(function, dimension, structure)

    state = tuple(Parser._namespace.keys())
    namespace = Parser(debug).parse_latex(sentence)
    if not isinstance(namespace, dict):
        return namespace
    if not namespace: return None

    frame = currentframe().f_back
    for key in namespace:
        if isinstance(namespace[key], IndexedSymbol):
            frame.f_globals[key] = namespace[key].structure
        elif isinstance(namespace[key], Function('Constant')):
            frame.f_globals[key] = namespace[key].args[0]

    overridden = [key for key in state if key in namespace]
    return tuple(('*' if symbol in overridden else '')
        + str(symbol) for symbol in namespace.keys())

class ParseOutput(tuple):
    """ Output Structure for IPython (Jupyter) """

    def __init__(self, iterable, sentence):
        self.iterable = iterable
        self.sentence = sentence

    def __new__(cls, iterable, sentence):
        return super(ParseOutput, cls).__new__(cls, iterable)

    def __eq__(self, other):
        return self.iterable == other.iterable and \
            self.sentence == other.sentence

    def __ne__(self, other):
        return self.iterable != other.iterable and \
            self.sentence != other.sentence

    def _repr_latex_(self):
        return r'\[' + self.sentence + r'\]'
