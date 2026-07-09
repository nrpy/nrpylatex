from typing import Any, Dict, Iterator, List, Optional, Union

from sympy import Expr, Function, Symbol

from .core.parser import Parser
from .utils.exceptions import NamespaceError
from .utils.structures import IndexedSymbol


class ParsedNamespace:
    def __init__(self, variables: Dict[str, Any], overridden: List[str]) -> None:
        self._variables = variables
        self._overridden = overridden
        for key, value in variables.items():
            setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        for symbol in self._variables:
            yield ('*' if symbol in self._overridden else '') + str(symbol)

    def __repr__(self) -> str:
        return f'ParsedNamespace({", ".join(self._variables.keys())})'


def parse_latex(
    sentence: str,
    reset: bool = False,
    debug: bool = False,
    namespace: Optional[Dict[str, Any]] = None,
) -> Union[Expr, ParsedNamespace]:
    if reset:
        Parser.initialize(reset=True)

    if namespace:
        for symbol, structure in namespace.items():
            function = Function('Tensor')(Symbol(symbol, real=True))
            if not isinstance(structure, list):
                raise NamespaceError(f'cannot import variable of type {type(structure)}, only list')
            dimension = len(structure)
            i = 0
            while i < len(structure) and isinstance(structure[i], list):
                if len(structure[i]) != dimension:
                    raise NamespaceError(f"inconsistent dimension in '{symbol}'")
                i += 1
            Parser._namespace[symbol] = IndexedSymbol(function, dimension, structure)

    state = tuple(Parser._namespace.keys())
    parsed_result = Parser(debug).parse_latex(sentence)

    if not isinstance(parsed_result, dict):
        return parsed_result

    extracted_vars: Dict[str, Any] = {}
    for key, value in parsed_result.items():
        if isinstance(value, IndexedSymbol):
            extracted_vars[key] = value.structure
        elif isinstance(value, Function('Constant')):
            extracted_vars[key] = value.args[0]
        else:
            extracted_vars[key] = value

    overridden = [key for key in state if key in parsed_result]
    return ParsedNamespace(extracted_vars, overridden)
