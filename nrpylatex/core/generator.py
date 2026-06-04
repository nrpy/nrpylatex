import math
import re
from itertools import chain
from typing import Any

from sympy import Add, Derivative, Function, Symbol, srepr

from ..utils.exceptions import NRPyLaTeXError
from ..utils.structures import ExprTree, IndexedSymbol


class Generator:
    def __init__(self, parser: Any) -> None:
        self._namespace: dict[str, Any] = parser._namespace
        self._property: dict[str, Any] = parser._property

    def generate(
        self, LHS: Any, RHS: Any, impsum: bool = True
    ) -> tuple[dict[str, Any], int | None, str | None]:
        # perform implied summation on indexed expression
        LHS_RHS, dimension, suffix = self.expand_summation(LHS, RHS, impsum)
        if self._property['debug']:
            lineno = f'[{self._property["debug"]}]'
            indent = len(lineno) * ' '
            print(f'{indent} \033[92mPython\033[00m')
            lhs_rhs_str = LHS_RHS.replace('\n', '\n      ')
            print(f'{indent}   {lhs_rhs_str}\n')
            self._property['debug'] += 1

        global_env = dict(self._namespace)
        for key in global_env:
            if isinstance(global_env[key], IndexedSymbol):
                global_env[key] = global_env[key].structure
            if isinstance(global_env[key], Function('Constant')):
                global_env[key] = global_env[key].args[0]
        global_env['coord'] = self._property['coord']

        # evaluate every implied summation and update namespace
        exec('from sympy import *', global_env)
        try:
            exec(LHS_RHS, global_env)
        except IndexError:
            raise GeneratorError('index out of range; change loop/summation range')

        return global_env, dimension, suffix

    def expand_summation(
        self, LHS: Any, RHS: Any, impsum: bool = True
    ) -> tuple[str, int | None, str | None]:
        tree, indexing = ExprTree(LHS), []
        for subtree in tree.preorder():
            subexpr = subtree.expr
            if subexpr.func == Function('Tensor'):
                for index, position in IndexedSymbol.indexing(subexpr):
                    if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                        indexing.append((index, position))
            elif subexpr.func == Derivative:
                for index, _ in subexpr.args[1:]:
                    if index not in self._property['coord']:
                        if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                            indexing.append((index, 'D'))

        symbol_LHS = IndexedSymbol(LHS).symbol
        # construct a tuple list of every LHS free index
        free_index_LHS = (
            self.separate_indexing(indexing, symbol_LHS, impsum)[0]
            if impsum
            else list(dict.fromkeys([(str(idx), pos) for idx, pos in indexing]))
        )
        # construct a tuple list of every RHS free index
        free_index_RHS: list[Any] = []

        iterable = RHS.args if RHS.func == Add else [RHS]
        LHS, RHS = IndexedSymbol(LHS).array_format(LHS), srepr(RHS)
        for element in iterable:
            index_range = self._property['index'].copy()
            original = srepr(element)
            if original[0] == '-':
                original = original[1:]
            modified = original
            indexing = []
            tree = ExprTree(element)

            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func == Function('Tensor'):
                    symbol = str(subexpr.args[0])
                    for index in subexpr.args[1:]:
                        if str(index) in self._property['index']:
                            dimension = self._property['index'][str(index)]
                        else:
                            dimension = self._namespace[symbol].dimension
                        if str(index) in index_range and dimension != index_range[str(index)]:
                            raise GeneratorError(
                                f"inconsistent loop/summation range for index '{index}'"
                            )
                        index_range[str(index)] = dimension
                    function = IndexedSymbol(subexpr).array_format(subexpr)
                    modified = modified.replace(srepr(subexpr), function)
                    for index, position in IndexedSymbol.indexing(subexpr):
                        if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                            indexing.append((index, position))
                elif subexpr.func == Function('Constant'):
                    constant = str(subexpr.args[0])
                    modified = modified.replace(srepr(subexpr), constant)
                elif subexpr.func == Derivative:
                    argument = subexpr.args[0]
                    derivative = f'diff({srepr(argument)}'
                    symbol = str(argument.args[0])
                    for index, order in subexpr.args[1:]:
                        if str(index) in self._property['index']:
                            dimension = self._property['index'][str(index)]
                        else:
                            dimension = self._namespace[symbol].dimension
                        if str(index) in index_range and dimension != index_range[str(index)]:
                            raise GeneratorError(
                                f"inconsistent loop/summation range for index '{index}'"
                            )
                        index_range[str(index)] = dimension
                        if index not in self._property['coord']:
                            derivative += f', (coord[{index}], {order})'
                            if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                                indexing.append((index, 'D'))
                        else:
                            derivative += f', ({index}, {order})'
                    derivative += ')'
                    modified = modified.replace(srepr(subexpr), derivative)
                    tmp = srepr(subexpr).replace(
                        srepr(argument), IndexedSymbol(argument).array_format(argument)
                    )
                    modified = modified.replace(tmp, derivative)

            if impsum:
                free_index, bound_index = self.separate_indexing(indexing, symbol_LHS, impsum)
                free_index_RHS.append(free_index)
                # generate implied summation over every bound index
                for idx in bound_index:
                    modified = f'sum({modified} for {idx} in range({index_range[idx]}))'
            else:
                free_index_RHS.append(indexing)
            RHS = RHS.replace(original, modified)

        if impsum:
            unique_LHS = list(dict.fromkeys([(str(idx), pos) for idx, pos in free_index_LHS]))
            unique_RHS = list(
                dict.fromkeys([(str(idx), pos) for idx, pos in chain(*free_index_RHS)])
            )
            for idx, pos in unique_LHS + unique_RHS:
                if ((idx, pos) in unique_LHS) != ((idx, pos) in unique_RHS):
                    # raise exception upon violation of the following rule:
                    # a free index must appear in every term with the same
                    # position and cannot be summed over in any term
                    raise GeneratorError(f"unbalanced free index '{idx}' in {symbol_LHS}")
        else:
            unique_LHS_str = list(dict.fromkeys([str(idx) for idx, _ in free_index_LHS]))
            unique_RHS_str = list(dict.fromkeys([str(idx) for idx, _ in chain(*free_index_RHS)]))
            for idx in unique_RHS_str:
                if idx not in unique_LHS_str:
                    # raise exception upon violation of the following rule:
                    # every index on the RHS must appear at least once on
                    # the LHS with the noimpsum annotation applied
                    raise GeneratorError(f"unbalanced index '{idx}' in {symbol_LHS}")

        # generate tensor instantiation with implied summation
        if symbol_LHS in self._namespace:
            equation_str = len(free_index_LHS) * '    ' + f'{LHS} = {RHS}'
            for i, (idx, _) in enumerate(reversed(free_index_LHS)):
                indent_level = len(free_index_LHS) - (i + 1)
                equation_str = (
                    indent_level * '    '
                    + f'for {idx} in range({index_range[idx]}):\n'
                    + equation_str
                )
            equation = [equation_str]
        else:
            for idx, _ in reversed(free_index_LHS):
                RHS = f'[{RHS} for {idx} in range({index_range[idx]})]'
            equation = [LHS.split('[')[0], RHS]

        dimension_LHS = None
        if free_index_LHS:
            if len(list(dict.fromkeys(index_range[index] for index, _ in free_index_LHS))) > 1:
                raise GeneratorError(f"cannot infer dimension of '{symbol_LHS}'")
            index, _ = free_index_LHS[0]
            dimension_LHS = index_range[index]

        # shift tensor indexing forward whenever dimension > upper bound
        # and infer derivative suffix of LHS tensor from RHS tensors
        suffix_LHS = None
        for subtree in tree.preorder():
            subexpr = subtree.expr
            if subexpr.func == Function('Tensor'):
                symbol = str(subexpr.args[0])
                dimension = self._namespace[symbol].dimension
                suffix = self._namespace[symbol].suffix
                if suffix is not None:
                    suffix_LHS = self._property['suffix']
                tensor = IndexedSymbol(subexpr, dimension)
                indexing = IndexedSymbol.indexing(subexpr)
                for index in subexpr.args[1:]:
                    if str(index) in self._property['index']:
                        upper_bound = self._property['index'][str(index)]
                        if dimension > upper_bound:
                            shift = dimension - upper_bound
                            for i, (idx, pos) in enumerate(indexing):
                                if str(idx) == str(index):
                                    indexing[i] = (f'{idx} + {shift}', pos)
                equation[-1] = equation[-1].replace(
                    tensor.array_format(subexpr), tensor.array_format(indexing)
                )

        return ' = '.join(equation), dimension_LHS, suffix_LHS

    @staticmethod
    def separate_indexing(
        indexing: list[tuple[Any, Any]], symbol_LHS: str, impsum: bool = True
    ) -> tuple[list[tuple[str, str]], list[str]]:
        free_index: list[tuple[str, str]] = []
        bound_index: list[str] = []
        str_indexing = [(str(idx), pos) for idx, pos in indexing]
        # iterate over every unique index in the subexpression
        for index in list(dict.fromkeys([idx for idx, _ in str_indexing])):
            count = U = D = 0
            index_tuple = []
            # count index occurrence and position occurrence
            for index_, position in str_indexing:
                if index_ == index:
                    index_tuple.append((index_, position))
                    if position == 'U':
                        U += 1
                    if position == 'D':
                        D += 1
                    count += 1
            # identify every bound index on the RHS
            if count > 1:
                if impsum and (count != 2 or U != D):
                    # raise exception upon violation of the following rule:
                    # a bound index must appear exactly once as a superscript
                    # and exactly once as a subscript in any single term
                    raise GeneratorError(f"illegal bound index '{index}' in {symbol_LHS}")
                bound_index.append(index)
            # identify every free index on the RHS
            else:
                free_index.extend(index_tuple)
        return list(dict.fromkeys(free_index)), bound_index

    @staticmethod
    def generate_metric(symbol: str, dimension: int, suffix: str | None) -> str:
        latex_config = ''
        sym_base = symbol[:-2]
        fact = math.factorial(dimension - 1)

        if 'U' in symbol:
            prefix = (
                r'\epsilon_{'
                + ' '.join(f'i_{i}' for i in range(1, 1 + dimension))
                + '} '
                + r'\epsilon_{'
                + ' '.join(f'j_{i}' for i in range(1, 1 + dimension))
                + '} '
            )
            det_latex = prefix + ' '.join(
                rf'\mathrm{{{sym_base}}}^{{i_{i} j_{i}}}' for i in range(1, 1 + dimension)
            )
            inv_latex = prefix + ' '.join(
                rf'\mathrm{{{sym_base}}}^{{i_{i} j_{i}}}' for i in range(2, 1 + dimension)
            )
            latex_config += f'% declare {sym_base}det --dim {dimension}'
            latex_config += rf"""
\mathrm{{{sym_base}det}} = \frac{{1}}{{({dimension})({fact})}} {det_latex} \\
\mathrm{{{sym_base}}}_{{i_1 j_1}} = \frac{{1}}{{{fact}}} \mathrm{{{sym_base}det}}^{{-1}} ({inv_latex}) \\"""
        else:
            prefix = (
                r'\epsilon^{'
                + ' '.join(f'i_{i}' for i in range(1, 1 + dimension))
                + '} '
                + r'\epsilon^{'
                + ' '.join(f'j_{i}' for i in range(1, 1 + dimension))
                + '} '
            )
            det_latex = prefix + ' '.join(
                rf'\mathrm{{{sym_base}}}_{{i_{i} j_{i}}}' for i in range(1, 1 + dimension)
            )
            inv_latex = prefix + ' '.join(
                rf'\mathrm{{{sym_base}}}_{{i_{i} j_{i}}}' for i in range(2, 1 + dimension)
            )
            latex_config += f'% declare {sym_base}det --dim {dimension}'
            latex_config += rf"""
\mathrm{{{sym_base}det}} = \frac{{1}}{{({dimension})({fact})}} {det_latex} \\
\mathrm{{{sym_base}}}^{{i_1 j_1}} = \frac{{1}}{{{fact}}} \mathrm{{{sym_base}det}}^{{-1}} ({inv_latex}) \\"""
        return latex_config

    @staticmethod
    def generate_connection(symbol: str, diacritic: str | None) -> str:
        metric = rf'\mathrm{{{symbol.rstrip("UD")}}}'
        diac_str = diacritic if diacritic else ''
        return rf'\mathrm{{Gamma{diac_str}}}^{{i_1}}_{{i_2 i_3}} = \frac{{1}}{{2}} {metric}^{{i_1 i_4}} (\partial_{{i_2}} {metric}_{{i_3 i_4}} + \partial_{{i_3}} {metric}_{{i_4 i_2}} - \partial_{{i_4}} {metric}_{{i_2 i_3}})'

    @staticmethod
    def generate_covdrv(
        function: Function,
        covdrv_index: Any,
        symbol: str | None = None,
        diacritic: str | None = None,
        dimension: int | None = None,
    ) -> str:
        indexing = [str(index) for index in function.args[1:]] + [str(covdrv_index)]
        idx_gen = IndexedSymbol.index_count()
        for i, index in enumerate(indexing):
            if index in indexing[:i]:
                indexing[i] = next(x for x in idx_gen if x not in indexing)
        covdrv_index = indexing[-1]
        if '_' in str(covdrv_index):
            base, subscript = str(covdrv_index).split('_')
            if len(base) > 1:
                covdrv_index = rf'\{base}_{subscript}'
        elif len(str(covdrv_index)) > 1:
            covdrv_index = rf'\{covdrv_index}'

        latex = IndexedSymbol.latex_format(
            Function('Tensor')(function.args[0], *(Symbol(i) for i in indexing[:-1]))
        )

        diac_str = rf'\{diacritic}{{\nabla}}' if diacritic else r'\nabla'
        LHS = rf'{diac_str}_{{{covdrv_index}}} {latex}'
        RHS = rf'\partial_{{{covdrv_index}}} {latex}'

        for index, (_, position) in zip(indexing, IndexedSymbol.indexing(function)):
            idx_gen = IndexedSymbol.index_count()
            bound_index = next(x for x in idx_gen if x not in indexing)
            latex = IndexedSymbol.latex_format(
                Function('Tensor')(
                    function.args[0],
                    *(Symbol(bound_index) if i == index else Symbol(i) for i in indexing[:-1]),
                )
            )
            if '_' in str(index):
                base, subscript = str(index).split('_')
                if len(base) > 1:
                    index = rf'\{base}_{subscript}'
            elif len(str(index)) > 1:
                index = rf'\{index}'

            RHS += ' + ' if position == 'U' else ' - '
            RHS += rf'\{diacritic}{{\mathrm{{Gamma}}}}' if diacritic else r'\mathrm{Gamma}'
            if position == 'U':
                RHS += rf'^{{{index}}}_{{{bound_index} {covdrv_index}}} ({latex})'
            else:
                RHS += rf'^{{{bound_index}}}_{{{index} {covdrv_index}}} ({latex})'
        return f'{LHS} = {RHS}'

    @staticmethod
    def generate_liedrv(function: Function, vector: Any, weight: Any = None) -> str:
        if len(str(vector)) > 1:
            vector = rf'\mathrm{{{vector}}}'
        indexing = [str(index) for index, _ in IndexedSymbol.indexing(function)]
        idx_gen = IndexedSymbol.index_count()
        for i, index in enumerate(indexing):
            if index in indexing[:i]:
                indexing[i] = next(x for x in idx_gen if x not in indexing)

        latex = IndexedSymbol.latex_format(function)
        LHS = rf'\mathcal{{L}}_{vector} {latex}'
        bound_index = next(x for x in idx_gen if x not in indexing)
        RHS = rf'{vector}^{{{bound_index}}} \partial_{{{bound_index}}} {latex}'

        for index, position in IndexedSymbol.indexing(function):
            latex = IndexedSymbol.latex_format(
                Function('Tensor')(
                    function.args[0],
                    *(Symbol(bound_index) if i == str(index) else Symbol(i) for i in indexing),
                )
            )
            if '_' in str(index):
                base, subscript = str(index).split('_')
                if len(base) > 1:
                    index = rf'\{base}_{subscript}'
            elif len(str(index)) > 1:
                index = rf'\{index}'

            if position == 'U':
                RHS += rf' - (\partial_{{{bound_index}}} {vector}^{{{index}}}) {latex}'
            else:
                RHS += rf' + (\partial_{{{index}}} {vector}^{{{bound_index}}}) {latex}'

        if weight:
            latex = IndexedSymbol.latex_format(function)
            RHS += rf' + ({weight})(\partial_{{{bound_index}}} {vector}^{{{bound_index}}}) {latex}'
        return f'{LHS} = {RHS}'


class GeneratorError(NRPyLaTeXError):
    def __init__(
        self, message: str, sentence: str | None = None, position: int | None = None
    ) -> None:
        super().__init__(message, sentence, position)
