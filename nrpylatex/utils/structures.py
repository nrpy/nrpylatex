"""Expression Trees, Coordinate Systems and Indexed Symbols"""

import re
import sys
from itertools import product
from typing import Any, Iterator, List, Optional, SupportsIndex, Tuple, Union, overload

from sympy import Function, Symbol, sympify


class ExprTree:
    """Symbolic Expression Tree

    >>> from sympy.abc import a, b, x
    >>> from sympy import cos
    >>> tree = ExprTree(cos(a + b)**2)
    >>> print(tree)
    ExprTree(cos(a + b)**2)
    >>> [node.expr for node in tree.preorder()]
    [cos(a + b)**2, cos(a + b), a + b, a, b, 2]
    """

    def __init__(self, expr: Any) -> None:
        self.root = self.Node(expr, None)
        self.build(self.root)

    def build(self, node: 'ExprTree.Node', clear: bool = True) -> None:
        """Build expression (sub)tree.

        :arg:   root node of (sub)tree
        :arg:   clear children (default: True)

        >>> from sympy.abc import a, b
        >>> from sympy import cos, sin
        >>> tree = ExprTree(cos(a + b)**2)
        >>> tree.root.expr = sin(a*b)**2
        >>> tree.build(tree.root, clear=True)
        >>> [node.expr for node in tree.preorder()]
        [sin(a*b)**2, sin(a*b), a*b, a, b, 2]
        """
        if clear:
            del node.children[:]
        for arg in node.expr.args:
            subtree = self.Node(arg, node.expr.func)
            node.append(subtree)
            self.build(subtree)

    def preorder(self, node: Optional['ExprTree.Node'] = None) -> Iterator['ExprTree.Node']:
        """Generate iterator for preorder traversal.

        :arg:    root node of (sub)tree
        :return: iterator

        >>> from sympy.abc import a, b
        >>> from sympy import cos, Mul
        >>> tree = ExprTree(cos(a*b)**2)
        >>> for i, subtree in enumerate(tree.preorder()):
        ...     if subtree.expr.func == Mul:
        ...         print((i, subtree.expr))
        (2, a*b)
        """
        if node is None:
            node = self.root
        yield node
        for child in node.children:
            for subtree in self.preorder(child):
                yield subtree

    def postorder(self, node: Optional['ExprTree.Node'] = None) -> Iterator['ExprTree.Node']:
        """Generate iterator for postorder traversal.

        :arg:    root node of (sub)tree
        :return: iterator

        >>> from sympy.abc import a, b
        >>> from sympy import cos, Mul
        >>> tree = ExprTree(cos(a*b)**2)
        >>> for i, subtree in enumerate(tree.postorder()):
        ...     if subtree.expr.func == Mul:
        ...         print((i, subtree.expr))
        (2, a*b)
        """
        if node is None:
            node = self.root
        for child in node.children:
            for subtree in self.postorder(child):
                yield subtree
        yield node

    def reconstruct(self, evaluate: bool = False) -> Any:
        """
        Reconstruct root expression from expression tree.

        :arg:    evaluate root expression (default: False)
        :return: root expression

        >>> from sympy.abc import a, b
        >>> from sympy import cos, sin
        >>> tree = ExprTree(cos(a + b)**2)
        >>> tree.root.children[0].expr = sin(a + b)
        >>> tree.reconstruct()
        sin(a + b)**2
        """
        for subtree in self.postorder():
            if subtree.children:
                expr_list = [node.expr for node in subtree.children]
                subtree.expr = subtree.expr.func(*expr_list, evaluate=evaluate)
        return self.root.expr

    class Node:
        """Expression Tree Node"""

        def __init__(self, expr: Any, func: Any) -> None:
            self.expr = expr
            self.func = func
            self.children: List['ExprTree.Node'] = []

        def append(self, node: 'ExprTree.Node') -> None:
            self.children.append(node)

        def __repr__(self) -> str:
            return f'Node({self.expr}, {self.func})'

        def __str__(self) -> str:
            return str(self.expr)

    def __repr__(self) -> str:
        return f'ExprTree({self.root.expr})'

    __str__ = __repr__


class CoordinateSystem(list[Symbol]):
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def default(self, n: int) -> Symbol:
        return Symbol(f'{self.symbol}_{n}', real=True)

    def index(self, value: Any, *args: Any, **kwargs: Any) -> int:
        pattern = re.match(f'{self.symbol}_([0-9][0-9]*)', str(value))
        if pattern is not None:
            return int(pattern.group(1))  # Cast to int to match supertype return
        return list.index(self, value, *args, **kwargs)

    def __missing__(self, index: int) -> Symbol:
        return self.default(index)

    @overload
    def __getitem__(self, i: SupportsIndex) -> Symbol: ...

    @overload
    def __getitem__(self, s: slice) -> List[Symbol]: ...

    def __getitem__(self, i: Union[SupportsIndex, slice]) -> Union[Symbol, List[Symbol]]:
        try:
            return super().__getitem__(i)
        except IndexError:
            if not isinstance(i, slice):
                return self.__missing__(int(i))
            raise

    def __contains__(self, key: Any) -> bool:
        return list.__contains__(self, key) or bool(
            re.match(f'{self.symbol}_[0-9][0-9]*', str(key))
        )

    def __eq__(self, other: Any) -> bool:
        return list.__eq__(self, other) and self.symbol == other.symbol


class IndexedSymbol:
    def __init__(
        self,
        function: Any,
        dimension: Optional[int] = None,
        structure: Optional[Any] = None,
        equation: Optional[Any] = None,
        symmetry: Optional[str] = None,
        suffix: Optional[str] = None,
        weight: Optional[Any] = None,
        impsum: bool = True,
    ) -> None:
        self.overridden = False
        self.symbol = str(function.args[0])
        self.rank = 0
        for symbol in re.split(r'_d[^UD]*|_cd|_ld', self.symbol):
            for character in reversed(symbol):
                if character in ('U', 'D'):
                    self.rank += 1
                else:
                    break
        self.dimension = dimension
        self.structure = structure
        self.equation = equation
        self.symmetry = symmetry
        self.suffix = suffix
        self.weight = weight
        self.impsum = impsum

    @staticmethod
    def indexing(function: Any) -> List[Tuple[Any, str]]:
        """Symbol Indexing from SymPy Function"""
        symbol, indices = function.args[0], function.args[1:]
        i, indexing = len(indices) - 1, []
        for symbol_part in reversed(re.split(r'_d[^UD]*|_cd|_ld', str(symbol))):
            for character in reversed(symbol_part):
                if character in ('U', 'D'):
                    indexing.append((indices[i], character))
                else:
                    break
                i -= 1
        return list(reversed(indexing))

    # TODO change method type to static (class) method
    def array_format(self, function: Any) -> str:
        """Indexed Symbol Notation for Array Formatting"""
        if isinstance(function, Function('Tensor')):
            indexing = self.indexing(function)
        else:
            indexing = function
        if not indexing:
            return self.symbol
        indices_str = ''.join([f'[{index}]' for index, _ in indexing])
        return f'{self.symbol}{indices_str}'

    @staticmethod
    def latex_format(function: Any) -> str:
        """Indexed Symbol Notation for LaTeX Formatting"""
        symbol, indexing = str(function.args[0]), IndexedSymbol.indexing(function)
        operator, i_2 = '', len(symbol)
        for i_1 in range(len(symbol), 0, -1):
            subsym = symbol[i_1:i_2]
            if '_d' in subsym:
                suffix = re.split(r'_d[^UD]*', subsym)[-1]
                for _ in reversed(suffix):
                    index = str(indexing.pop()[0])
                    if '_' in index:
                        base, subscript = index.split('_')
                        if len(base) > 1:
                            index = f'\\{base}_{{{subscript}}}'
                    elif len(index) > 1:
                        index = f'\\{index}'
                    operator += f'\\partial_{{{index}}} '
                i_2 = i_1
            elif '_cd' in subsym:
                suffix = subsym.split('_cd')[-1]
                diacritic = (
                    'bar'
                    if 'bar' in suffix
                    else 'hat'
                    if 'hat' in suffix
                    else 'tilde'
                    if 'tilde' in suffix
                    else None
                )
                if diacritic:
                    suffix = suffix[len(diacritic) :]
                for position in reversed(suffix):
                    index = str(indexing.pop()[0])
                    if '_' in index:
                        base, subscript = index.split('_')
                        if len(base) > 1:
                            index = f'\\{base}_{{{subscript}}}'
                    elif len(index) > 1:
                        index = f'\\{index}'
                    operator += f'\\{diacritic}{{\\nabla}}' if diacritic else '\\nabla'
                    if position == 'U':
                        operator += f'^{{{index}}}'
                    else:
                        operator += f'_{{{index}}}'
                    operator += ' '
                i_2 = i_1
            elif '_ld' in subsym:
                vector = re.split('_ld', subsym)[-1]
                if len(vector) > 1:
                    vector = f'\\mathrm{{{vector}}}'
                operator += f'\\mathcal{{L}}_{vector} '
                i_2 = i_1
        symbol = re.split(r'_d[^UD]*|_cd|_ld', symbol)[0]
        for i, character in enumerate(reversed(symbol)):
            if character not in ('U', 'D'):
                symbol = symbol[: len(symbol) - i]
                break

        latex_0 = symbol
        latex_1_list: List[str] = []
        latex_2_list: List[str] = []

        if len(latex_0) > 1:
            latex_0 = f'\\mathrm{{{latex_0}}}'
        latex_0 = f'{operator}{latex_0}'

        U_count, D_count = 0, 0
        for index_val, position in indexing:
            index_str = str(index_val)
            if '_' in index_str:
                base, subscript = index_str.split('_')
                if len(base) > 1:
                    index_str = f'\\{base}_{{{subscript}}}'
            elif len(index_str) > 1:
                index_str = f'\\{index_str}'

            if position == 'U':
                latex_1_list.append(index_str)
                U_count += 1
            else:
                latex_2_list.append(index_str)
                D_count += 1

        latex_1 = ' '.join(latex_1_list)
        latex_2 = ' '.join(latex_2_list)

        if U_count > 0:
            latex_1 = f'^{{{latex_1}}}'
        if D_count > 0:
            latex_2 = f'_{{{latex_2}}}'

        return f'{latex_0}{latex_1}{latex_2}'

    @staticmethod
    def index_count() -> Iterator[str]:
        n = 1
        while True:
            yield f'i_{n}'
            n += 1

    def __repr__(self) -> str:
        symbol = ('*' if self.overridden else '') + self.symbol
        if self.rank == 0:
            return f'Scalar({symbol})'
        return f'Tensor({symbol}, {self.dimension}D)'

    __str__ = __repr__


class IndexedSymbolError(Exception):
    """Invalid Indexed Symbol"""


def symdef(
    rank: int,
    symbol: Optional[str] = None,
    symmetry: Optional[str] = None,
    dimension: Optional[int] = None,
) -> Any:
    """Generate an indexed symbol of specified rank and dimension

    >>> indexed_symbol =  symdef(rank=2, symbol='M', dimension=3, symmetry='sym01')
    >>> assert pipe(indexed_symbol, lambda x: repeat(flatten, x, 1), set, len) == 6

    >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='sym01')
    >>> assert len(set(repeat(flatten, indexed_symbol, 2))) == 18
    >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='sym02')
    >>> assert len(set(repeat(flatten, indexed_symbol, 2))) == 18
    >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='sym12')
    >>> assert len(set(repeat(flatten, indexed_symbol, 2))) == 18

    >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='sym012')
    >>> assert len(set(repeat(flatten, indexed_symbol, 2))) == 10

    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym01')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym02')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym03')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym12')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym13')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym23')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54

    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym012')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 30
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym013')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 30
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym01_sym23')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 36
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym02_sym13')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 36
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym023')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 30
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym03_sym12')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 36
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym123')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 30

    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym0123')
    >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 15

    >>> indexed_symbol =  symdef(rank=2, symbol='M', dimension=3, symmetry='anti01')
    >>> assert len(set(map(abs, repeat(flatten, indexed_symbol, 1))).difference({0})) == 3
    >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='anti012')
    >>> assert len(set(map(abs, repeat(flatten, indexed_symbol, 2))).difference({0})) == 1
    >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='anti0123')
    >>> assert len(set(map(abs, repeat(flatten, indexed_symbol, 3))).difference({0})) == 0
    """
    if not dimension or dimension == -1:
        dimension = 3
    if symbol is not None:
        if not isinstance(symbol, str) or not re.match(r'[\w_]', symbol):
            raise ValueError('symbol must be an alphabetic string')
    if dimension is not None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError('dimension must be a positive integer')
    indexed_symbol = _init(rank * [dimension], symbol)
    if symmetry:
        return _symmetrize(rank, indexed_symbol, symmetry, dimension)
    return indexed_symbol


def _init(
    shape: Union[int, List[int]], symbol: Optional[str], index: Optional[List[int]] = None
) -> List[Any]:
    if isinstance(shape, int):
        shape = [shape]
    if not index:
        index = []
    iterable = [
        Symbol(f'{symbol}{"".join(str(n) for n in index + [i])}') if symbol else sympify(0)
        for i in range(shape[0])
    ]
    if len(shape) > 1:
        for i in range(shape[0]):
            iterable[i] = _init(shape[1:], symbol, index + [i])
    return iterable


def _symmetrize(rank: int, indexed_symbol: List[Any], symmetry: str, dimension: int) -> List[Any]:
    if rank == 1:
        if symmetry == 'nosym':
            return indexed_symbol
        raise IndexedSymbolError('cannot symmetrize indexed symbol of rank 1')
    if rank == 2:
        indexed_symbol = _symmetrize_rank2(indexed_symbol, symmetry, dimension)
    elif rank == 3:
        indexed_symbol = _symmetrize_rank3(indexed_symbol, symmetry, dimension)
    elif rank == 4:
        indexed_symbol = _symmetrize_rank4(indexed_symbol, symmetry, dimension)
    else:
        raise IndexedSymbolError('unsupported rank for indexed symbol')
    return indexed_symbol


def _symmetrize_rank2(indexed_symbol: List[Any], symmetry: str, dimension: int) -> List[Any]:
    for sym in symmetry.split('_'):
        sign = 1 if sym[:3] == 'sym' else -1
        for i, j in product(range(dimension), repeat=2):
            if sym[-2:] == '01':
                if j < i:
                    indexed_symbol[i][j] = sign * indexed_symbol[j][i]
                elif i == j and sign < 0:
                    indexed_symbol[i][j] = sympify(0)
            elif sym == 'nosym':
                pass
            else:
                raise IndexedSymbolError(f"unsupported symmetry option '{sym}'")
    return indexed_symbol


def _symmetrize_rank3(indexed_symbol: List[Any], symmetry: str, dimension: int) -> List[Any]:
    symmetry_str = symmetry
    symmetry_list: List[str] = []
    for sym in symmetry_str.split('_'):
        index = 3 if sym[:3] == 'sym' else 4
        if len(sym[index:]) == 3:
            prefix = sym[:index]
            symmetry_list.append(f'{prefix}{sym[index : (index + 2)]}')
            symmetry_list.append(f'{prefix}{sym[(index + 1) : (index + 3)]}')
        else:
            symmetry_list.append(sym)
    for sym in (symmetry_list[k] for n in range(len(symmetry_list), 0, -1) for k in range(n)):
        sign = 1 if sym[:3] == 'sym' else -1
        for i, j, k in product(range(dimension), repeat=3):
            if sym[-2:] == '01':
                if j < i:
                    indexed_symbol[i][j][k] = sign * indexed_symbol[j][i][k]
                elif i == j and sign < 0:
                    indexed_symbol[i][j][k] = sympify(0)
            elif sym[-2:] == '02':
                if k < i:
                    indexed_symbol[i][j][k] = sign * indexed_symbol[k][j][i]
                elif i == k and sign < 0:
                    indexed_symbol[i][j][k] = sympify(0)
            elif sym[-2:] == '12':
                if k < j:
                    indexed_symbol[i][j][k] = sign * indexed_symbol[i][k][j]
                elif j == k and sign < 0:
                    indexed_symbol[i][j][k] = sympify(0)
            elif sym == 'nosym':
                pass
            else:
                raise IndexedSymbolError(f"unsupported symmetry option '{sym}'")
    return indexed_symbol


def _symmetrize_rank4(indexed_symbol: List[Any], symmetry: str, dimension: int) -> List[Any]:
    symmetry_str = symmetry
    symmetry_list: List[str] = []
    for sym in symmetry_str.split('_'):
        index = 3 if sym[:3] == 'sym' else 4
        if len(sym[index:]) in (3, 4):
            prefix = sym[:index]
            symmetry_list.append(f'{prefix}{sym[index : (index + 2)]}')
            symmetry_list.append(f'{prefix}{sym[(index + 1) : (index + 3)]}')
            if len(sym[index:]) == 4:
                symmetry_list.append(f'{prefix}{sym[(index + 2) : (index + 4)]}')
        else:
            symmetry_list.append(sym)
    for sym in (symmetry_list[k] for n in range(len(symmetry_list), 0, -1) for k in range(n)):
        sign = 1 if sym[:3] == 'sym' else -1
        for i, j, k, m in product(range(dimension), repeat=4):
            if sym[-2:] == '01':
                if j < i:
                    indexed_symbol[i][j][k][m] = sign * indexed_symbol[j][i][k][m]
                elif i == j and sign < 0:
                    indexed_symbol[i][j][k][m] = sympify(0)
            elif sym[-2:] == '02':
                if k < i:
                    indexed_symbol[i][j][k][m] = sign * indexed_symbol[k][j][i][m]
                elif i == k and sign < 0:
                    indexed_symbol[i][j][k][m] = sympify(0)
            elif sym[-2:] == '03':
                if m < i:
                    indexed_symbol[i][j][k][m] = sign * indexed_symbol[m][j][k][i]
                elif i == m and sign < 0:
                    indexed_symbol[i][j][k][m] = sympify(0)
            elif sym[-2:] == '12':
                if k < j:
                    indexed_symbol[i][j][k][m] = sign * indexed_symbol[i][k][j][m]
                elif j == k and sign < 0:
                    indexed_symbol[i][j][k][m] = sympify(0)
            elif sym[-2:] == '13':
                if m < j:
                    indexed_symbol[i][j][k][m] = sign * indexed_symbol[i][m][k][j]
                elif j == m and sign < 0:
                    indexed_symbol[i][j][k][m] = sympify(0)
            elif sym[-2:] == '23':
                if m < k:
                    indexed_symbol[i][j][k][m] = sign * indexed_symbol[i][j][m][k]
                elif k == m and sign < 0:
                    indexed_symbol[i][j][k][m] = sympify(0)
            elif sym == 'nosym':
                pass
            else:
                raise IndexedSymbolError(f"unsupported symmetry option '{sym}'")
    return indexed_symbol


if __name__ == '__main__':
    import doctest

    sys.exit(doctest.testmod()[0])
