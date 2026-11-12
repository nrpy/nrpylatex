from __future__ import annotations

from collections.abc import Iterator
from inspect import currentframe
from types import TracebackType
from typing import TYPE_CHECKING, Any

from sympy import Expr, Function, Symbol

from .core.parser import Parser
from .utils.exceptions import NamespaceError
from .utils.structures import IndexedSymbol

if TYPE_CHECKING:
    from typing import Self

_MISSING: object = object()


class ParsedNamespace:
    def __init__(
        self,
        extracted_vars: dict[str, Any],
        overridden_keys: list[str],
        target_globals: dict[str, Any] | None = None,
        backup_globals: dict[str, Any] | None = None,
    ) -> None:
        self._extracted_vars = extracted_vars
        self._overridden_keys = overridden_keys
        self._target_globals = target_globals
        self._backup_globals = backup_globals
        for key, value in extracted_vars.items():
            setattr(self, key, value)

    def __enter__(self) -> Self:
        if self._target_globals is not None and self._backup_globals is None:
            self._backup_globals = {
                key: self._target_globals.get(key, _MISSING) for key in self._extracted_vars
            }
            self._target_globals.update(self._extracted_vars)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._target_globals is not None and self._backup_globals is not None:
            for key, orig_value in self._backup_globals.items():
                if orig_value is _MISSING:
                    self._target_globals.pop(key, None)
                else:
                    self._target_globals[key] = orig_value
            self._backup_globals = None

    def __iter__(self) -> Iterator[str]:
        for symbol in self._extracted_vars:
            yield ('*' if symbol in self._overridden_keys else '') + str(symbol)

    def __repr__(self) -> str:
        return f'ParsedNamespace({", ".join(self)})'


def parse_latex(
    sentence: str,
    namespace: dict[str, Any] | None = None,
    *,
    reset: bool = False,
    debug: bool = False,
    inject: bool = True,
) -> Expr | ParsedNamespace:
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

    extracted_vars: dict[str, Any] = {}
    for key, value in parsed_result.items():
        if isinstance(value, IndexedSymbol):
            extracted_vars[key] = value.structure
        elif isinstance(value, Function('Constant')):
            extracted_vars[key] = value.args[0]
        else:
            extracted_vars[key] = value

    frame = currentframe()
    target_globals = frame.f_back.f_globals if frame and frame.f_back else None
    del frame

    backup_globals: dict[str, Any] | None = None
    if target_globals is not None and inject:
        backup_globals = {key: target_globals.get(key, _MISSING) for key in extracted_vars}
        target_globals.update(extracted_vars)

    overridden_keys = [key for key in state if key in parsed_result]
    return ParsedNamespace(extracted_vars, overridden_keys, target_globals, backup_globals)
