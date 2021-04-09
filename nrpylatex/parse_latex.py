""" NRPyLaTeX: Convert LaTeX Sentence to SymPy Expression """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

# pylint: disable = attribute-defined-outside-init, protected-access, exec-used
from sympy import Function, Derivative, Symbol, Integer, Rational, Float, Pow, Add, Mul
from sympy import sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh
from sympy import pi, exp, log, sqrt, expand, diff, srepr
from inspect import currentframe
from nrpylatex.indexed_symbol import symdef
from nrpylatex.functional import uniquify
from nrpylatex.symtree import ExprTree
import re, sys, math, warnings

class Lexer:
    """ LaTeX Lexer

        The following class will tokenize a LaTeX sentence for parsing.
    """

    def __init__(self):
        # define a regex pattern for every token, create a named capture group for
        # every pattern, join together the resulting pattern list using a pipe symbol
        # for regex alternation, and compile the generated regular expression
        symmetry = r'nosym|(?:sym|anti)[0-9]+(?:_(?:sym|anti)[0-9]+)*'
        alphabet = '|'.join(letter for letter in (r'\\[aA]lpha', r'\\[bB]eta', r'\\[gG]amma', r'\\[dD]elta',
            r'\\[eE]psilon', r'\\[zZ]eta', r'\\[eE]ta', r'\\[tT]heta', r'\\[iI]ota', r'\\[kK]appa', r'\\[lL]ambda',
            r'\\[mM]u', r'\\[nN]u', r'\\[xX]i', r'\\[oO]mikron', r'\\[pP]i', r'\\[Rr]ho', r'\\[sS]igma', r'\\[tT]au',
            r'\\[uU]psilon', r'\\[pP]hi', r'\\[cC]hi', r'\\[pP]si', r'\\[oO]mega', r'[a-zA-Z]'))
        self.token_dict = [
            ('WHITESPACE',      r'\s+'),
            ('STRING',          r'\"[^\"]*\"'),
            ('GROUP',           r'\<[0-9]+(\.{2})?\>'),
            ('DIMENSION',       r'[2-9][0-9]*D'),
            ('VARIABLE',        r'\'[a-zA-Z][_a-zA-Z]*\''),
            ('RATIONAL',        r'\-?[0-9]+\/\-?[1-9][0-9]*'),
            ('DECIMAL',         r'\-?[0-9]+\.[0-9]+'),
            ('INTEGER',         r'\-?[0-9]+'),
            ('ARROW',           r'\-\>'),
            ('PLUS',            r'\+'),
            ('MINUS',           r'\-'),
            ('DIVIDE',          r'\/'),
            ('EQUAL',           r'\='),
            ('CARET',           r'\^'),
            ('UNDERSCORE',      r'\_'),
            ('COMMENT',         r'\%'),
            ('PRIME',           r'\''),
            ('COMMA',           r'\,'),
            ('COLON',           r'\:'),
            ('SEMICOLON',       r'\;'),
            ('LPAREN',          r'\('),
            ('RPAREN',          r'\)'),
            ('LBRACK',          r'\['),
            ('RBRACK',          r'\]'),
            ('LBRACE',          r'\{'),
            ('RBRACE',          r'\}'),
            ('OPENING',         r'\\begin{align\*?}'),
            ('CLOSING',         r'\\end{align\*?}'),
            ('PAR_SYM',         r'\\partial'),
            ('COV_SYM',         r'\\nabla'),
            ('LIE_SYM',         r'\\mathcal\{L\}'),
            ('TEXT_CMD',        r'\\text'),
            ('FUNC_CMD',        r'\\exp'),
            ('FRAC_CMD',        r'\\frac'),
            ('SQRT_CMD',        r'\\sqrt'),
            ('NLOG_CMD',        r'\\ln|\\log'),
            ('TRIG_CMD',        r'\\sinh|\\cosh|\\tanh|\\sin|\\cos|\\tan'),
            ('VARDEF_MACRO',    r'vardef'),
            ('KEYDEF_MACRO',    r'keydef'),
            ('ASSIGN_MACRO',    r'assign'),
            ('IGNORE_MACRO',    r'ignore'),
            ('PARSE_MACRO',     r'parse'),
            ('SREPL_MACRO',     r'srepl'),
            ('INDEX_KWRD',      r'index'),
            ('BASIS_KWRD',      r'basis'),
            ('DIFF_TYPE',       r'diff_type'),
            ('CONSTANT',        r'const'),
            ('KRONECKER',       r'kron'),
            ('METRIC',          r'metric'),
            ('SYMMETRY',        r'symmetry'),
            ('WEIGHT',          r'weight'),
            ('ZERO',            r'zero'),
            ('PERSIST',         r'persist'),
            ('DIFF_OPT',        r'symbolic|dD|dupD'),
            ('SYM_OPT',         symmetry),
            ('DIACRITIC',       r'\\hat|\\tilde|\\bar'),
            ('VPHANTOM',        r'\\vphantom'),
            ('PI',              r'\\pi'),
            ('LETTER',          r'[a-zA-Z]|' + alphabet),
            ('COMMAND',         r'\\[a-zA-Z]+'),
            ('RETURN',          r'\\{2}'),
            ('ESCAPE',          r'\\')]
        self.regex = re.compile('|'.join(['(?P<%s>%s)' % pattern for pattern in self.token_dict]))
        self.token_dict = dict(self.token_dict)

    def initialize(self, sentence, position=0):
        """ Initialize Lexer

            :arg: sentence (raw string)
            :arg: position [default: 0]
        """
        self.sentence = sentence
        self.token    = None
        self.lexeme   = None
        self.marker   = None
        self.index    = position

    def tokenize(self):
        """ Tokenize Sentence

            :return: token iterator
        """
        while self.index < len(self.sentence):
            token = self.regex.match(self.sentence, self.index)
            if token is None:
                raise ParseError('unexpected \'%s\' at position %d' %
                    (self.sentence[self.index], self.index), self.sentence, self.index)
            self.index = token.end()
            if token.lastgroup != 'WHITESPACE':
                self.lexeme = token.group()
                yield token.lastgroup

    def lex(self):
        """ Retrieve Next Token

            :return: next token
        """
        try:
            self.token = next(self.tokenize())
        except StopIteration:
            self.token  = None
            self.lexeme = ''
        return self.token

    def mark(self):
        """ Mark Iterator Position

            :return: previous position
        """
        self.marker = self.index - len(self.lexeme)
        return self.marker

    def reset(self):
        """ Reset Token Iterator """
        if not self.sentence:
            raise RuntimeError('cannot reset uninitialized lexer')
        self.initialize(self.sentence, self.marker)
        self.lex()

class Parser:
    """ LaTeX Parser

        The following class will parse a tokenized LaTeX sentence.
    """
        # LaTeX Extended BNF Grammar:
        # <LATEX>         -> ( <ALIGN> | '%' <MACRO> | <ASSIGNMENT> ) { [ <RETURN> ] ( <ALIGN> | '%' <MACRO> | <ASSIGNMENT> ) }*
        # <ALIGN>         -> <OPENING> ( '%' <MACRO> | <ASSIGNMENT> ) { [ <RETURN> ] ( '%' <MACRO> | <ASSIGNMENT> ) }* <CLOSING>
        #     <MACRO>     -> <PARSE> | <SREPL> | <VARDEF> | <KEYDEF> | <ASSIGN> | <IGNORE>
        #     <PARSE>     -> <PARSE_MACRO> <ASSIGNMENT> { ',' <ASSIGNMENT> }*
        #     <SREPL>     -> <SREPL_MACRO> [ '-' <PERSIST> ] <STRING> <ARROW> <STRING> { ',' <STRING> <ARROW> <STRING> }*
        #     <VARDEF>    -> <VARDEF_MACRO> { '-' ( <OPTION> | <ZERO> ) }* <VARIABLE> { ',' <VARIABLE> }* [ '(' <DIMENSION> ')' ]
        #     <KEYDEF>    -> <KEYDEF_MACRO> <BASIS_KWRD> <BASIS> | <INDEX_KWRD> <INDEX>
        #     <ASSIGN>    -> <ASSIGN_MACRO> { '-' <OPTION> }* <VARIABLE> { ',' <VARIABLE> }*
        #     <IGNORE>    -> <IGNORE_MACRO> <STRING> { ',' <STRING> }*
        #     <OPTION>    -> <CONSTANT> | <KRONECKER> | <METRIC> [ '=' <VARIABLE> ] | <WEIGHT> '=' <NUMBER>
        #                     | <DIFF_TYPE> '=' <DIFF_OPT> | <SYMMETRY> '=' <SYM_OPT>
        #     <BASIS>     -> <BASIS_KWRD> ( '[' <LETTER> { ',' <LETTER> }* ']' )
        #     <INDEX>     -> ( <LETTER> | '[' <LETTER> '-' <LETTER> ']' ) '(' <DIMENSION> ')'
        # <ASSIGNMENT>    -> <OPERATOR> = <EXPRESSION>
        # <EXPRESSION>    -> <TERM> { ( '+' | '-' ) <TERM> }*
        # <TERM>          -> <FACTOR> { [ '/' ] <FACTOR> }*
        # <FACTOR>        -> <BASE> { '^' <EXPONENT> }*
        # <BASE>          -> [ '-' ] ( <NUMBER> | <COMMAND> | <OPERATOR> | <SUBEXPR> )
        # <EXPONENT>      -> <BASE> | '{' <EXPRESSION> '}' | '{' '{' <EXPRESSION> '}' '}'
        # <SUBEXPR>       -> '(' <EXPRESSION> ')' | '[' <EXPRESSION> ']' | '\' '{' <EXPRESSION> '\' '}'
        # <COMMAND>       -> <FUNC> | <FRAC> | <SQRT> | <NLOG> | <TRIG>
        # <FUNC>          -> <FUNC_CMD> <SUBEXPR>
        # <FRAC>          -> <FRAC_CMD> '{' <EXPRESSION> '}' '{' <EXPRESSION> '}'
        # <SQRT>          -> <SQRT_CMD> [ '[' <INTEGER> ']' ] '{' <EXPRESSION> '}'
        # <NLOG>          -> <NLOG_CMD> [ '_' ( <NUMBER> | '{' <NUMBER> '}' ) ] ( <NUMBER> | <TENSOR> | <SUBEXPR> )
        # <TRIG>          -> <TRIG_CMD> [ '^' ( <NUMBER> | '{' <NUMBER> '}' ) ] ( <NUMBER> | <TENSOR> | <SUBEXPR> )
        # <OPERATOR>      -> [ <VPHANTOM> '{' <DIFF_OPT> '}' ] ( <PARDRV> | <COVDRV> | <LIEDRV> | <TENSOR> )
        # <PARDRV>        -> <PAR_SYM> [ '^' <INTEGER> ] '_' <LETTER> ( <OPERATOR> | <SUBEXPR> )
        # <COVDRV>        -> ( <COV_SYM> | <DIACRITIC> '{' <COV_SYM> '}' ) ( '^' | '_' ) <LETTER> ( <OPERATOR> | <SUBEXPR> )
        # <LIEDRV>        -> <LIE_SYM> '_' <SYMBOL> ( <OPERATOR> | <SUBEXPR> )
        # <TENSOR>        -> <SYMBOL> [ ( '_' <LOWER_INDEX> ) | ( '^' <UPPER_INDEX> [ '_' <LOWER_INDEX> ] ) ]
        # <SYMBOL>        -> <LETTER> | <DIACRITIC> '{' <SYMBOL> '}' | <TEXT_CMD> '{' <LETTER> { '_' | <LETTER> | <INTEGER> }* '}'
        # <LOWER_INDEX>   -> <LETTER> | <INTEGER> | '{' { <LETTER> | <INTEGER> }* [ ( ',' | ';' ) { <LETTER> | <INTEGER> }+ ] '}'
        # <UPPER_INDEX>   -> <LETTER> | <INTEGER> | '{' { <LETTER> | <INTEGER> }* '}'
        # <NUMBER>        -> <RATIONAL> | <DECIMAL> | <INTEGER> | <PI>

    _namespace, _property = {}, {}
    continue_parsing = True

    def __init__(self, debug=False):
        self.lexer = Lexer()
        if not self._property:
            self._property['dimension'] = 3
            self._property['srepl'] = []
            self._property['basis'] = []
            self._property['index'] = {i: self._property['dimension']
                for i in (chr(i) for i in range(105, 123))} # TODO 105 -> 97
            self._property['ignore'] = ['\\left', '\\right', '{}', '&']
            self._property['metric'] = {'': 'g', 'bar': 'g', 'hat': 'g', 'tilde': 'gamma'}
        if 'vphantom' not in self._property:
            self._property['vphantom'] = None
        def excepthook(exception_type, exception, traceback):
            if not debug:
                # remove traceback from exception message
                print('%s: %s' % (exception_type.__name__, exception))
            else: sys.__excepthook__(exception_type, exception, traceback)
        sys.excepthook = excepthook

    def parse(self, sentence, expression=False):
        """ Parse LaTeX Sentence

            :arg:    latex sentence (raw string)
            :arg:    expression mode [default: disabled]
            :return: namespace or expression
        """
        # replace every substring marked 'ignore' with an empty string
        for ignore in self._property['ignore']:
            sentence = sentence.replace(ignore, '')
        # perform string replacement (aliasing) using namespace mapping
        self.lexer.initialize('\n'.join(['srepl "%s" -> "%s"' % (old, new)
            for (old, new) in self._property['srepl']] + [sentence]))
        self.lexer.lex()
        for _ in self._property['srepl']:
            self._srepl()
        sentence = self.lexer.sentence[self.lexer.mark():]
        stack = []; i = i_1 = i_2 = i_3 = 0
        while i < len(sentence):
            lexeme = sentence[i]
            if   lexeme == '(': stack.append(i)
            elif lexeme == ')':
                i_1, i_2 = stack.pop(), i + 1
                if i_2 < len(sentence) and sentence[i_2] != '_':
                    i_1 = i_2 = 0
            elif i_2 != 0:
                # replace comma notation with operator notation for parenthetical expression(s)
                if lexeme == ',' and sentence[i - 1] == '{':
                    i_3 = sentence.find('}', i) + 1
                    subexpr, indexing = sentence[i_1:i_2], sentence[i_2:i_3][3:-1]
                    indexing = reversed(re.findall(self.lexer.token_dict['LETTER'], indexing))
                    operator = ' '.join('\\partial_' + index for index in indexing)
                    sentence = sentence.replace(sentence[i_1:i_3], operator + ' ' + subexpr)
                    i = i_1 + len(operator + ' ' + subexpr) - 1
                # replace semicolon notation with operator notation for parenthetical expression(s)
                elif lexeme == ';' and sentence[i - 1] == '{':
                    i_3 = sentence.find('}', i) + 1
                    subexpr, indexing = sentence[i_1:i_2], sentence[i_2:i_3][3:-1]
                    indexing = reversed(re.findall(self.lexer.token_dict['LETTER'], indexing))
                    operator = ' '.join('\\nabla_' + index for index in indexing)
                    sentence = sentence.replace(sentence[i_1:i_3], operator + ' ' + subexpr)
                    i = i_1 + len(operator + ' ' + subexpr) - 1
            i += 1
        i = 0
        # replace every comment (%%...\n) with an empty string
        while i < len(sentence) - 1:
            if sentence[i:(i + 2)] == '%%':
                index = sentence.index('\n', i + 2)
                sentence = sentence.replace(sentence[i:index], '')
            else: i += 1
        self.lexer.initialize(sentence)
        self.lexer.lex()
        if expression:
            tree = ExprTree(self._expression())
            # remove wrapper function from every scalar quantity, excluding constant(s)
            for subtree in tree.preorder():
                subexpr, rank = subtree.expr, len(subtree.expr.args)
                if rank == 1 and subexpr.func == Function('Tensor'):
                    subtree.expr = subexpr.args[0]
                    del subtree.children[:]
            return tree.reconstruct()
        self._latex()
        return self._namespace

    # <LATEX> -> ( <ALIGN> | '%' <MACRO> | <ASSIGNMENT> ) { [ <RETURN> ] ( <ALIGN> | '%' <MACRO> | <ASSIGNMENT> ) }*
    def _latex(self):
        while self.lexer.lexeme:
            if self.peek('OPENING'):
                self._align()
                if self.lexer.lexeme: continue
            elif self.accept('COMMENT'):
                self._macro()
            else: self._assignment()
            if self.accept('RETURN'): pass

    # <ALIGN> -> <OPENING> ( '%' <MACRO> | <ASSIGNMENT> ) { [ <RETURN> ] ( '%' <MACRO> | <ASSIGNMENT> ) }* <CLOSING>
    def _align(self):
        self.expect('OPENING')
        while not self.accept('CLOSING'):
            if self.accept('COMMENT'):
                self._macro()
            else: self._assignment()
            if self.accept('RETURN'): pass

    # <MACRO> -> <PARSE> | <SREPL> | <VARDEF> | <KEYDEF> | <ASSIGN> | <IGNORE>
    def _macro(self):
        macro = self.lexer.lexeme
        if self.peek('PARSE_MACRO'):
            self._parse()
        elif self.peek('SREPL_MACRO'):
            self._srepl()
        elif self.peek('VARDEF_MACRO'):
            self._vardef()
        elif self.peek('KEYDEF_MACRO'):
            self._keydef()
        elif self.peek('ASSIGN_MACRO'):
            self._assign()
        elif self.peek('IGNORE_MACRO'):
            self._ignore()
        else:
            sentence, position = self.lexer.sentence, self.lexer.mark()
            raise ParseError('unsupported macro \'%s\' at position %d' %
                (macro, position), sentence, position)

    # <PARSE> -> <PARSE_MACRO> <ASSIGNMENT> { ',' <ASSIGNMENT> }*
    def _parse(self):
        self.expect('PARSE_MACRO')
        self._assignment()
        while self.accept('COMMA'):
            self._assignment()

    # <SREPL> -> <SREPL_MACRO> [ '-' <PERSIST> ] <STRING> <ARROW> <STRING> { ',' <STRING> <ARROW> <STRING> }*
    def _srepl(self):
        self.expect('SREPL_MACRO')
        persist = self.accept('MINUS') and self.accept('PERSIST')
        while True:
            old = self.lexer.lexeme[1:-1]
            self.expect('STRING')
            self.expect('ARROW')
            new = self.lexer.lexeme[1:-1]
            if persist and [old, new] not in self._property['srepl']:
                self._property['srepl'].append([old, new])
            self.lexer.mark()
            self.expect('STRING')
            self.lexer.reset(); self.lexer.mark()
            lexer = Lexer(); lexer.initialize(old)
            substr_syntax = []
            for token in lexer.tokenize():
                substr_syntax.append((lexer.lexeme, token))
            string_syntax = []
            for token in self.lexer.tokenize():
                string_syntax.append((self.lexer.index, self.lexer.lexeme, token))
            sentence = self.lexer.sentence
            i_1 = i_2 = offset = 0
            for i, (index, lexeme, token) in enumerate(string_syntax):
                if substr_syntax[0][0] == lexeme or substr_syntax[0][1] == 'GROUP' or token == 'TEXT_CMD':
                    k, index, varmap = i, index - len(lexeme), {}
                    for j, (_lexeme, _token) in enumerate(substr_syntax, start=i):
                        if k >= len(string_syntax): break
                        if _token == 'LETTER' and string_syntax[k][2] == 'TEXT_CMD':
                            letter_1 = _lexeme[1:] if len(_lexeme) > 1 else _lexeme
                            letter_2, l = string_syntax[k + 2][1], 2
                            while string_syntax[k + l + 1][2] != 'RBRACE':
                                letter_2 += string_syntax[k + l + 1][1]
                                l += 1
                            if letter_1 != letter_2: break
                            k += l + 1
                        elif _token == 'GROUP':
                            varmap[_lexeme] = string_syntax[k][1]
                            if _lexeme[-2] == '.':
                                l, string = k + 1, varmap[_lexeme]
                                if l < len(string_syntax) and j - i + 1 < len(substr_syntax):
                                    while string_syntax[l][1] != substr_syntax[j - i + 1][0]:
                                        string += string_syntax[l][1]
                                        if l + 1 >= len(string_syntax): break
                                        l += 1
                                    else:
                                        k, varmap[_lexeme] = l - 1, string
                        elif _lexeme != string_syntax[k][1]: break
                        if (j - i + 1) == len(substr_syntax):
                            new_repl = new
                            for var in varmap:
                                new_repl = new_repl.replace(var, varmap[var])
                            i_1, i_2 = index + offset, string_syntax[k][0] + offset
                            old_repl = sentence[i_1:i_2]
                            sentence = sentence[:i_1] + new_repl + sentence[i_2:]
                            offset += len(new_repl) - len(old_repl)
                        k += 1
            self.lexer.sentence = sentence
            self.lexer.reset(); self.lexer.lex()
            if not self.accept('COMMA'): break

    # <VARDEF> -> <VARDEF_MACRO> { '-' ( <OPTION> | <ZERO> ) }* <VARIABLE> { ',' <VARIABLE> }* [ '(' <DIMENSION> ')' ]
    def _vardef(self):
        self.expect('VARDEF_MACRO')
        diff_type, symmetry = None, None
        metric, weight = None, None
        zero = False
        while self.accept('MINUS'):
            zero = self.accept('ZERO')
            if zero: continue
            option, value = self._option().split('<>')
            if option == 'diff_type':
                diff_type = value
            elif option == 'symmetry':
                symmetry = value
            elif option == 'metric':
                metric = value
            elif option == 'weight':
                weight = value
        while True:
            symbol = self.lexer.lexeme[1:-1]
            if not symbol and self.peek('VARIABLE'):
                sentence, position = self.lexer.sentence, self.lexer.mark()
                raise ParseError('empty variable at position %d' %
                    position, sentence, position)
            self.expect('VARIABLE')
            dimension = self._property['dimension']
            if self.accept('LPAREN'):
                dimension = self.lexer.lexeme[:-1]
                self.expect('DIMENSION')
                dimension = int(dimension)
                self.expect('RPAREN')
            if symmetry == 'const':
                self._namespace[symbol] = Function('Constant')(Symbol(symbol, real=True))
            else:
                function = Function('Tensor')(Symbol(symbol, real=True))
                tensor = Tensor(function, dimension, diff_type=diff_type, metric=metric, weight=weight)
                if symmetry == 'kron':
                    if tensor.rank != 2:
                        raise TensorError('cannot instantiate kronecker delta of rank ' + str(tensor.rank))
                    tensor.structure = [[1 if i == j else 0 for j in range(dimension)] for i in range(dimension)]
                tensor.symmetry = ('sym01' if symmetry in ('kron', 'metric') else symmetry)
                self._define_tensor(tensor, zero=zero)
            if symmetry == 'metric':
                diacritic = next(diacritic for diacritic in ('bar', 'hat', 'tilde', '') if diacritic in symbol)
                self._property['metric'][diacritic] = re.split(diacritic if diacritic else r'[UD]', symbol)[0]
                christoffel = 'Gamma' + diacritic + 'UDD'
                if christoffel in self._namespace:
                    del self._namespace[christoffel]
                sentence, position = self.lexer.sentence, self.lexer.mark()
                self.parse(self._generate_metric(symbol, dimension, diff_type))
                self.lexer.initialize(sentence, position)
                self.lexer.lex()
            if not self.accept('COMMA'): break

    # <KEYDEF> -> <KEYDEF_MACRO> <BASIS_KWRD> <BASIS> | <INDEX_KWRD> <INDEX>
    def _keydef(self):
        self.expect('KEYDEF_MACRO')
        if self.accept('BASIS_KWRD'):
            self._basis()
        elif self.accept('INDEX_KWRD'):
            self._index()
        else:
            sentence, position = self.lexer.sentence, self.lexer.mark()
            raise ParseError('unexpected keyword at position %d' %
                position, sentence, position)

    # <ASSIGN> -> <ASSIGN_MACRO> { '-' <OPTION> }* <VARIABLE> { ',' <VARIABLE> }*
    def _assign(self):
        self.expect('ASSIGN_MACRO')
        diff_type, symmetry = None, None
        metric, weight = None, None
        while self.accept('MINUS'):
            option, value = self._option().split('<>')
            if option == 'diff_type':
                diff_type = value
            elif option == 'symmetry':
                symmetry = value
            elif option == 'metric':
                metric = value
            elif option == 'weight':
                weight = value
        while True:
            symbol = self.lexer.lexeme[1:-1]
            if not symbol and self.peek('VARIABLE'):
                sentence, position = self.lexer.sentence, self.lexer.mark()
                raise ParseError('empty variable at position %d' %
                    position, sentence, position)
            self.expect('VARIABLE')
            if symbol not in self._namespace:
                rank = 0
                for symbol in re.split(r'_d|_dup|_cd|_ld', symbol):
                    for character in reversed(symbol):
                        if character in ('U', 'D'):
                            rank += 1
                        else: break
                if rank != 0:
                    raise TensorError('cannot update undefined tensor \'' + symbol + '\'')
                dimension = self._property['dimension']
                function = Function('Tensor')(Symbol(symbol, real=True))
                self._define_tensor(Tensor(function, dimension, diff_type=diff_type))
            tensor = self._namespace[symbol]
            if diff_type:
                tensor.diff_type = diff_type
            else: diff_type = tensor.diff_type
            if symmetry:
                tensor.symmetry = symmetry
            else: symmetry = tensor.symmetry
            if metric:
                tensor.metric = metric
            if weight:
                tensor.weight = weight
            else: weight = tensor.weight
            dimension = tensor.dimension
            if symmetry == 'metric':
                if tensor.rank != 2:
                    raise TensorError('cannot assign -metric to \'{symbol}\' since rank({symbol}) != 2')
                metric = tensor.structure
                for i in range(tensor.dimension):
                    for j in range(tensor.dimension):
                        if metric[i][j] == 0:
                            metric[i][j] = metric[j][i]
                        elif metric[j][i] == 0:
                            metric[j][i] = metric[i][j]
                        if metric[i][j] != metric[j][i]:
                            raise TensorError('cannot assign -metric to \'{symbol}\' since {symbol}[{i}][{j}] != {symbol}[{j}][{i}]'
                                .format(symbol=symbol, i=i, j=j))
                symmetry = tensor.symmetry = 'sym01'
                diacritic = next(diacritic for diacritic in ('bar', 'hat', 'tilde', '') if diacritic in symbol)
                self._property['metric'][diacritic] = re.split(diacritic if diacritic else r'[UD]', symbol)[0]
                christoffel = 'Gamma' + diacritic + 'UDD'
                if christoffel in self._namespace:
                    del self._namespace[christoffel]
                sentence, position = self.lexer.sentence, self.lexer.mark()
                self.parse(self._generate_metric(symbol, dimension, diff_type))
                self.lexer.initialize(sentence, position)
                self.lexer.lex()
            base_symbol = re.split(r'_d|_dup|_cd|_ld', symbol)[0]
            if base_symbol and diff_type:
                rank = 0
                for symbol in re.split(r'_d|_dup|_cd|_ld', symbol):
                    for character in reversed(symbol):
                        if character in ('U', 'D'):
                            rank += 1
                        else: break
                if base_symbol in self._namespace:
                    self._namespace[base_symbol].diff_type = diff_type
                elif rank == 0:
                    function = Function('Tensor')(Symbol(base_symbol, real=True))
                    self._define_tensor(Tensor(function, diff_type=diff_type))
            if not self.accept('COMMA'): break

    # <IGNORE> -> <IGNORE_MACRO> <STRING> { ',' <STRING> }*
    def _ignore(self):
        self.expect('IGNORE_MACRO')
        while True:
            string = self.lexer.lexeme[1:-1]
            if len(string) > 0 and string not in self._property['ignore']:
                self._property['ignore'].append(string)
            sentence, position = self.lexer.sentence, self.lexer.index
            self.lexer.mark()
            self.expect('STRING')
            if len(string) > 0:
                self.lexer.sentence = sentence[:position] + sentence[position:].replace(string, '')
            if not self.accept('COMMA'): break
        self.lexer.reset(); self.lexer.lex()

    # <OPTION> -> <CONSTANT> | <KRONECKER> | <METRIC> [ '=' <VARIABLE> ] | <WEIGHT> '=' <NUMBER>
    #              | <DIFF_TYPE> '=' <DIFF_OPT> | <SYMMETRY> '=' <SYM_OPT>
    def _option(self):
        if self.accept('CONSTANT'):
            return 'symmetry<>const'
        if self.accept('KRONECKER'):
            return 'symmetry<>kron'
        if self.accept('METRIC'):
            if self.accept('EQUAL'):
                metric = self.lexer.lexeme[1:-1]
                self.expect('VARIABLE')
                return 'metric<>' + metric
            return 'symmetry<>metric'
        if self.accept('WEIGHT'):
            self.expect('EQUAL')
            weight = self._number()
            return 'weight<>' + weight
        if self.accept('DIFF_TYPE'):
            self.expect('EQUAL')
            diff_type = self.lexer.lexeme
            self.lexer.lex()
            return 'diff_type<>' + diff_type
        if self.accept('SYMMETRY'):
            self.expect('EQUAL')
            symmetry = self.lexer.lexeme
            self.lexer.lex()
            return 'symmetry<>' + symmetry
        sentence, position = self.lexer.sentence, self.lexer.mark()
        raise ParseError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <BASIS> -> <BASIS_KWRD> <LBRACK> <LETTER> [ ',' <LETTER> ]* <RBRACK>
    def _basis(self):
        self.expect('LBRACK')
        del self._property['basis'][:]
        while True:
            symbol = self._strip(self.lexer.lexeme)
            self.expect('LETTER')
            if symbol in self._property['basis']:
                sentence, position = self.lexer.sentence, self.lexer.mark()
                raise ParseError('duplicate basis symbol \'%s\' at position %d' %
                    (sentence[position], position), sentence, position)
            self._property['basis'].append(Symbol(symbol, real=True))
            if not self.accept('COMMA'): break
        self.expect('RBRACK')
        if not self._property['dimension']:
            self._property['dimension'] = len(self._property['basis'])

    # <INDEX> -> ( <LETTER> | '[' <LETTER> '-' <LETTER> ']' )  '(' <DIMENSION> ')'
    def _index(self):
        if self.accept('LBRACK'):
            index_1 = self._strip(self.lexer.lexeme)
            self.expect('LETTER')
            self.expect('MINUS')
            index_2 = self._strip(self.lexer.lexeme)
            self.expect('LETTER')
            index = [chr(i) for i in range(ord(index_1), ord(index_2) + 1)]
            self.expect('RBRACK')
        else:
            index = [self._strip(self.lexer.lexeme)]
            self.expect('LETTER')
        self.expect('LPAREN')
        dimension = self.lexer.lexeme
        self.expect('DIMENSION')
        dimension = int(dimension[:-1])
        self.expect('RPAREN')
        self._property['index'].update({i: dimension for i in index})

    # <ASSIGNMENT> -> <OPERATOR> = <EXPRESSION>
    def _assignment(self):
        function = self._operator('LHS')
        indexed = function.func == Function('Tensor') and len(function.args) > 1
        self.expect('EQUAL')
        sentence, position = self.lexer.sentence, self.lexer.mark()
        tree = ExprTree(self._expression())
        equation = ((Tensor.latex_format(function), sentence[position:self.lexer.mark()]), tree.root.expr)
        if not indexed:
            for subtree in tree.preorder():
                subexpr, rank = subtree.expr, len(subtree.expr.args)
                if subexpr.func == Function('Tensor') and rank > 1:
                    indexed = True
        LHS, RHS = function, expand(tree.root.expr) if indexed else tree.root.expr
        # perform implied summation on indexed expression
        LHS_RHS, dimension = self._summation(LHS, RHS)
        global_env = self._namespace.copy()
        global_env.update(self._property)
        exec('from sympy import *', global_env)
        for key in global_env:
            if isinstance(global_env[key], Tensor):
                global_env[key] = global_env[key].structure
            if isinstance(global_env[key], Function('Constant')):
                global_env[key] = global_env[key].args[0]
        # evaluate every implied summation and update namespace
        exec(LHS_RHS, global_env)
        symbol, indices = str(function.args[0]), function.args[1:]
        if any(isinstance(index, Integer) for index in indices):
            tensor = self._namespace[symbol]
            tensor.structure = global_env[symbol]
        else:
            diff_type = self._namespace[symbol].diff_type if symbol in self._namespace else None
            tensor = Tensor(function, dimension, structure=global_env[symbol],
                equation=equation, diff_type=diff_type)
        self._namespace.update({symbol: tensor})

    # <EXPRESSION> -> <TERM> { ( '+' | '-' ) <TERM> }*
    def _expression(self):
        expr = self._term()
        while self.peek('PLUS') or self.peek('MINUS'):
            if self.accept('PLUS'):
                expr += self._term()
            elif self.accept('MINUS'):
                expr -= self._term()
        return expr

    # <TERM> -> <FACTOR> { [ '/' ] <FACTOR> }*
    def _term(self):
        expr = self._factor()
        while any(self.peek(token) for token in ('DIVIDE',
                'RATIONAL', 'DECIMAL', 'INTEGER', 'PI', 'PAR_SYM', 'COV_SYM', 'LIE_SYM',
                'TEXT_CMD', 'FUNC_CMD', 'FRAC_CMD', 'SQRT_CMD', 'NLOG_CMD', 'TRIG_CMD',
                'LPAREN', 'LBRACK', 'DIACRITIC', 'VPHANTOM', 'LETTER', 'COMMAND', 'ESCAPE')):
            self.lexer.mark()
            if self.accept('ESCAPE'):
                if self.peek('RBRACE'):
                    self.lexer.reset()
                    return expr
                self.lexer.reset()
            if self.accept('DIVIDE'):
                expr /= self._factor()
            else: expr *= self._factor()
        return expr

    # <FACTOR> -> <BASE> { '^' <EXPONENT> }*
    def _factor(self):
        stack = [self._base()]
        while self.accept('CARET'):
            stack.append(self._exponent())
        if len(stack) == 1: stack.append(1)
        expr = stack.pop()
        for subexpr in reversed(stack):
            exponential = (subexpr == Function('Tensor')(Symbol('e', real=True)))
            expr = exp(expr) if exponential else subexpr ** expr
        return expr

    # <BASE> -> [ '-' ] ( <NUMBER> | <COMMAND> | <OPERATOR> | <SUBEXPR> )
    def _base(self):
        sign = -1 if self.accept('MINUS') else 1
        if any(self.peek(token) for token in
                ('RATIONAL', 'DECIMAL', 'INTEGER', 'PI')):
            return sign * self._number()
        if any(self.peek(token) for token in
                ('FUNC_CMD', 'FRAC_CMD', 'SQRT_CMD', 'NLOG_CMD', 'TRIG_CMD', 'COMMAND')):
            return sign * self._command()
        if any(self.peek(token) for token in
                ('VPHANTOM', 'DIACRITIC', 'PAR_SYM', 'COV_SYM', 'LIE_SYM', 'LETTER', 'TEXT_CMD')):
            return sign * self._operator()
        if any(self.peek(i) for i in ('LPAREN', 'LBRACK', 'ESCAPE')):
            return sign * self._subexpr()
        sentence, position = self.lexer.sentence, self.lexer.mark()
        raise ParseError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <EXPONENT> -> <BASE> | '{' <EXPRESSION> '}' | '{' '{' <EXPRESSION> '}' '}'
    def _exponent(self):
        if self.accept('LBRACE'):
            if self.accept('LBRACE'):
                base = self._expression()
                self.expect('RBRACE')
            else: base = self._expression()
            self.expect('RBRACE')
            return base
        return self._base()

    # <SUBEXPR> -> '(' <EXPRESSION> ')' | '[' <EXPRESSION> ']' | '\' '{' <EXPRESSION> '\' '}'
    def _subexpr(self):
        if self.accept('LPAREN'):
            expr = self._expression()
            self.expect('RPAREN')
        elif self.accept('LBRACK'):
            expr = self._expression()
            self.expect('RBRACK')
        elif self.accept('ESCAPE'):
            self.expect('LBRACE')
            expr = self._expression()
            self.expect('ESCAPE')
            self.expect('RBRACE')
        else:
            sentence, position = self.lexer.sentence, self.lexer.mark()
            raise ParseError('unexpected \'%s\' at position %d' %
                (sentence[position], position), sentence, position)
        return expr

    # <COMMAND> -> <FUNC> | <FRAC> | <SQRT> | <NLOG> | <TRIG>
    def _command(self):
        command = self.lexer.lexeme
        if self.peek('FUNC_CMD'):
            return self._func()
        if self.peek('FRAC_CMD'):
            return self._frac()
        if self.peek('SQRT_CMD'):
            return self._sqrt()
        if self.peek('NLOG_CMD'):
            return self._nlog()
        if self.peek('TRIG_CMD'):
            return self._trig()
        sentence, position = self.lexer.sentence, self.lexer.mark()
        raise ParseError('unsupported command \'%s\' at position %d' %
            (command, position), sentence, position)

    # <FUNC> -> <FUNC_CMD> '(' <EXPRESSION> ')'
    def _func(self):
        func = self._strip(self.lexer.lexeme)
        self.expect('FUNC_CMD')
        self.expect('LPAREN')
        expr = self._expression()
        self.expect('RPAREN')
        if func == 'exp':
            return exp(expr)
        sentence, position = self.lexer.sentence, self.lexer.mark()
        raise ParseError('unsupported function \'%s\' at position %d' %
            (func, position), sentence, position)

    # <FRAC> -> <FRAC_CMD> '{' <EXPRESSION> '}' '{' <EXPRESSION> '}'
    def _frac(self):
        self.expect('FRAC_CMD')
        self.expect('LBRACE')
        numerator = self._expression()
        self.expect('RBRACE')
        self.expect('LBRACE')
        denominator = self._expression()
        self.expect('RBRACE')
        return numerator / denominator

    # <SQRT> -> <SQRT_CMD> [ '[' <INTEGER> ']' ] '{' <EXPRESSION> '}'
    def _sqrt(self):
        self.expect('SQRT_CMD')
        if self.accept('LBRACK'):
            integer = self.lexer.lexeme
            self.expect('INTEGER')
            root = Rational(1, integer)
            self.expect('RBRACK')
        else: root = Rational(1, 2)
        self.expect('LBRACE')
        expr = self._expression()
        self.expect('RBRACE')
        if root == Rational(1, 2):
            return sqrt(expr)
        return Pow(expr, root)

    # <NLOG> -> <NLOG_CMD> [ '_' ( <NUMBER> | '{' <NUMBER> '}' ) ] ( <NUMBER> | <TENSOR> | <SUBEXPR> )
    def _nlog(self):
        func = self._strip(self.lexer.lexeme)
        self.expect('NLOG_CMD')
        if func == 'log':
            if self.accept('UNDERSCORE'):
                if self.accept('LBRACE'):
                    base = self._number()
                    self.expect('RBRACE')
                else:
                    base = self._number()
                base = int(base)
            else: base = 10
        if any(self.peek(token) for token in
                ('RATIONAL', 'DECIMAL', 'INTEGER', 'PI')):
            expr = self._number()
        elif any(self.peek(token) for token in
                ('LETTER', 'DIACRITIC', 'TEXT_CMD')):
            expr = self._tensor()
        elif any(self.peek(i) for i in ('LPAREN', 'LBRACK', 'LBRACE')):
            expr = self._subexpr()
        else:
            sentence, position = self.lexer.sentence, self.lexer.mark()
            raise ParseError('unexpected \'%s\' at position %d' %
                (sentence[position], position), sentence, position)
        if func == 'ln': return log(expr)
        return log(expr, base)

    # <TRIG> -> <TRIG_CMD> [ '^' ( <NUMBER> | '{' <NUMBER> '}' ) ] ( <NUMBER> | <TENSOR> | <SUBEXPR> )
    def _trig(self):
        func = self._strip(self.lexer.lexeme)
        self.expect('TRIG_CMD')
        if self.accept('CARET'):
            if self.accept('LBRACE'):
                exponent = self._number()
                self.expect('RBRACE')
            else:
                exponent = self._number()
            exponent = int(exponent)
        else: exponent = 1
        if   func == 'cosh': trig = acosh if exponent == -1 else cosh
        elif func == 'sinh': trig = asinh if exponent == -1 else sinh
        elif func == 'tanh': trig = atanh if exponent == -1 else tanh
        elif func == 'cos':  trig = acos  if exponent == -1 else cos
        elif func == 'sin':  trig = asin  if exponent == -1 else sin
        elif func == 'tan':  trig = atan  if exponent == -1 else tan
        if any(self.peek(token) for token in
                ('RATIONAL', 'DECIMAL', 'INTEGER', 'PI')):
            expr = self._number()
        elif any(self.peek(token) for token in
                ('LETTER', 'DIACRITIC', 'TEXT_CMD')):
            expr = self._tensor()
        elif any(self.peek(i) for i in ('LPAREN', 'LBRACK', 'LBRACE')):
            expr = self._subexpr()
        else:
            sentence, position = self.lexer.sentence, self.lexer.mark()
            raise ParseError('unexpected \'%s\' at position %d' %
                (sentence[position], position), sentence, position)
        if exponent == -1: return trig(expr)
        return trig(expr) ** exponent

    # <OPERATOR> -> [ <VPHANTOM> '{' <DIFF_OPT> '}' ] ( <PARDRV> | <COVDRV> | <LIEDRV> | <TENSOR> )
    def _operator(self, location='RHS'):
        vphantom = self._property['vphantom']
        if self.accept('VPHANTOM'):
            self.expect('LBRACE')
            diff_type = self.lexer.lexeme
            self.expect('DIFF_OPT')
            self._property['vphantom'] = diff_type
            self.expect('RBRACE')
        if not vphantom and location == 'LHS':
            self._property['vphantom'] = 'dD'
        operator = self.lexer.lexeme
        if self.peek('PAR_SYM'):
            pardrv = self._pardrv(location)
            self._property['vphantom'] = vphantom
            return pardrv
        if self.peek('COV_SYM') or self.peek('DIACRITIC') or \
                (self.peek('LETTER') and self.lexer.lexeme == 'D'):
            self.lexer.mark()
            if self.accept('DIACRITIC'):
                self.expect('LBRACE')
                if self.peek('COV_SYM') or (self.peek('LETTER') and self.lexer.lexeme == 'D'):
                    self.lexer.reset()
                    covdrv = self._covdrv(location)
                    self._property['vphantom'] = vphantom
                    return covdrv
                self.lexer.reset()
            else:
                covdrv = self._covdrv(location)
                self._property['vphantom'] = vphantom
                return covdrv
        if self.peek('LIE_SYM'):
            liedrv = self._liedrv(location)
            self._property['vphantom'] = vphantom
            return liedrv
        if any(self.peek(token) for token in ('LETTER', 'DIACRITIC', 'TEXT_CMD')):
            tensor = self._tensor(location)
            self._property['vphantom'] = vphantom
            return tensor
        sentence, position = self.lexer.sentence, self.lexer.mark()
        raise ParseError('unsupported operator \'%s\' at position %d' %
            (operator, position), sentence, position)

    # <PARDRV> -> <PAR_SYM> [ '^' <INTEGER> ] '_' <LETTER> ( <OPERATOR> | <SUBEXPR> )
    def _pardrv(self, location='RHS'):
        self.expect('PAR_SYM')
        if self.accept('CARET'):
            order = self.lexer.lexeme
            self.expect('INTEGER')
            order = int(order)
        else: order = 1
        self.expect('UNDERSCORE')
        index = self._strip(self.lexer.lexeme)
        self.expect('LETTER')
        index = Symbol(index, real=True)
        if any(self.peek(i) for i in ('LPAREN', 'LBRACK', 'LBRACE')):
            subexpr = self._subexpr()
            if order > 1 or index in self._property['basis']:
                return Derivative(subexpr, (index, order))
            tree = ExprTree(subexpr)
            # insert temporary symbol '_x' for symbolic differentiation
            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func in (Function('Tensor'), Derivative):
                    subtree.expr = Function('Function')(subexpr, Symbol('_x'))
                    del subtree.children[:]
            expr = tree.reconstruct()
            # differentiate the expression, including product rule expansion
            tree = ExprTree(diff(expr, (Symbol('_x'), order)))
            # remove temporary symbol '_x' from tensor function
            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func == Derivative:
                    function = subexpr.args[0].args[0]
                    if function.func == Derivative:
                        subtree.expr = Derivative(function, index)
                    else:
                        symbol = str(function.args[0])
                        tensor = self._namespace[symbol]
                        vphantom = self._property['vphantom']
                        diff_type = tensor.diff_type
                        if vphantom and diff_type != 'symbolic':
                            diff_type = vphantom
                        if index in self._property['basis']:
                            diff_type = 'symbolic'
                        subtree.expr = self._define_pardrv(function, location, diff_type, index)
                    del subtree.children[:]
                elif subexpr.func == Function('Function'):
                    subtree.expr = subexpr.args[0]
                    del subtree.children[:]
            return tree.reconstruct()
        function = self._operator()
        if function.func == Derivative or index in self._property['basis']:
            return Derivative(function, (index, order))
        symbol = str(function.args[0])
        tensor = self._namespace[symbol]
        vphantom = self._property['vphantom']
        diff_type = tensor.diff_type
        if vphantom and diff_type != 'symbolic':
            diff_type = vphantom
        if index in self._property['basis']:
            diff_type = 'symbolic'
        return self._define_pardrv(function, location, diff_type, index)

    # <COVDRV> -> ( <COV_SYM> | <DIACRITIC> '{' <COV_SYM> '}' ) ( '^' | '_' ) <LETTER> ( <OPERATOR> | <SUBEXPR> )
    def _covdrv(self, location='RHS'):
        diacritic, position = '', self.lexer.mark()
        if self.peek('DIACRITIC'):
            diacritic = self._strip(self.lexer.lexeme)
            self.expect('DIACRITIC')
            operator = '\\' + diacritic + '{\\nabla}'
            self.expect('LBRACE')
            if self.peek('LETTER') and self.lexer.lexeme == 'D':
                self.lexer.lex()
            else: self.expect('COV_SYM')
            self.expect('RBRACE')
        else:
            operator = '\\nabla'
            if self.peek('LETTER') and self.lexer.lexeme == 'D':
                self.lexer.lex()
            else: self.expect('COV_SYM')
        metric = self._property['metric'][diacritic] + diacritic
        if metric + 'DD' not in self._namespace:
            raise ParseError('cannot generate covariant derivative without defined metric \'%s\'' %
                metric, self.lexer.sentence, position)
        if self.accept('CARET'):
            lexeme = self._strip(self.lexer.lexeme)
            self.expect('LETTER')
            index = (Symbol(lexeme, real=True), 'U')
        elif self.accept('UNDERSCORE'):
            lexeme = self._strip(self.lexer.lexeme)
            self.expect('LETTER')
            index = (Symbol(lexeme, real=True), 'D')
        else:
            sentence, position = self.lexer.sentence, self.lexer.mark()
            raise ParseError('unexpected \'%s\' at position %d' %
                (sentence[position], position), sentence, position)
        func_list, expression = self._expand_product(location, 'cd' + diacritic, index[1], index[0])
        for symbol, function in func_list:
            equation = [operator, ' = ', '', operator]
            alphabet = (chr(97 + n) for n in range(26))
            indexing = [str(i) for i in function.args[1:]] + [str(index[0])]
            for i, idx in enumerate(indexing):
                if idx in indexing[:i]:
                    indexing[i] = next(x for x in alphabet if x not in indexing)
            latex = Tensor.latex_format(Function('Tensor')(function.args[0],
                        *(Symbol(i) for i in indexing[:-1])))
            covdrv_index = indexing[-1]
            if index[1] == 'U':
                equation[0] += '^' + covdrv_index + ' '
                bound_index = next(x for x in alphabet if x not in indexing)
                prefix = '\\' if len(self._property['metric'][diacritic]) > 1 else ''
                metric = prefix + self._property['metric'][diacritic]
                if diacritic: metric = '\\%s{%s}' % (diacritic, metric)
                equation[2] += '%s^{%s %s} ' % (metric, covdrv_index, bound_index)
                equation[3] += '_' + bound_index + ' '
            else:
                equation[0] += '_' + covdrv_index + ' '
                equation[3] += '_' + covdrv_index + ' '
            equation[0], equation[3] = equation[0] + latex, equation[3] + latex
            if location == 'RHS' and (self._property['vphantom'] or symbol not in self._namespace):
                sentence, position = self.lexer.sentence, self.lexer.mark()
                if index[1] == 'U':
                    config = ' % assign -diff_type=dD \'' + symbol + '\''
                    self.parse(''.join(equation) + config)
                else:
                    self.parse(self._generate_covdrv(function, index[0], symbol, diacritic))
                self.lexer.initialize(sentence, position)
                self.lexer.lex()
        return expression

    # <LIEDRV> -> <LIE_SYM> '_' <SYMBOL> ( <OPERATOR> | <SUBEXPR> )
    def _liedrv(self, location='RHS'):
        self.expect('LIE_SYM')
        self.expect('UNDERSCORE')
        vector = self._strip(self._symbol())
        func_list, expression = self._expand_product(location, 'ld', vector)
        for symbol, function in func_list:
            if location == 'RHS' and (self._property['vphantom'] or symbol not in self._namespace):
                sentence, position = self.lexer.sentence, self.lexer.mark()
                symbol = str(function.args[0])
                tensor = Tensor(function, self._namespace[symbol].dimension)
                tensor.weight = self._namespace[symbol].weight
                self.parse(self._generate_liedrv(function, vector, tensor.weight))
                self.lexer.initialize(sentence, position)
                self.lexer.lex()
        return expression

    # <TENSOR> -> <SYMBOL> [ ( '_' <LOWER_INDEX> ) | ( '^' <UPPER_INDEX> [ '_' <LOWER_INDEX> ] ) ]
    def _tensor(self, location='RHS'):
        sentence, position = self.lexer.sentence, self.lexer.mark()
        indexing = []
        symbol = list(self._strip(self._symbol()))
        if self.accept('UNDERSCORE'):
            index, order, covariant = self._lower_index()
            indexing.extend(index)
            symbol.extend((len(index) - order) * ['D'])
            if order > 0:
                sentence = self.lexer.sentence
                suffix = '_cd' if covariant else '_d'
                symbol.append(suffix + order * 'D')
                function = Function('Tensor')(Symbol(''.join(symbol)), *indexing)
                old_latex = sentence[position:self.lexer.mark()]
                new_latex = Tensor(function).latex_format(function)
                self.lexer.sentence = sentence.replace(old_latex, new_latex)
                self.lexer.marker = position
                self.lexer.reset()
                return self._operator()
        self.lexer.mark()
        if self.accept('CARET'):
            if self.accept('LBRACE'):
                if self.accept('LBRACE'):
                    self.lexer.reset()
                    symbol = ''.join(symbol)
                    function = Function('Tensor')(Symbol(symbol, real=True))
                    if symbol in self._namespace:
                        if isinstance(self._namespace[symbol], Function('Constant')):
                            return self._namespace[symbol]
                    else: self._define_tensor(Tensor(function))
                    return function
                self.lexer.reset(); self.lexer.lex()
            index = self._upper_index()
            indexing.extend(index)
            symbol.extend(len(index) * ['U'])
            if self.accept('UNDERSCORE'):
                index, order, covariant = self._lower_index()
                indexing.extend(index)
                symbol.extend((len(index) - order) * ['D'])
                if order > 0:
                    sentence = self.lexer.sentence
                    suffix = '_cd' if covariant else '_d'
                    symbol.append(suffix + order * 'D')
                    function = Function('Tensor')(Symbol(''.join(symbol)), *indexing)
                    old_latex = sentence[position:self.lexer.mark()]
                    new_latex = Tensor(function).latex_format(function)
                    self.lexer.sentence = sentence.replace(old_latex, new_latex)
                    self.lexer.marker = position
                    self.lexer.reset()
                    return self._operator()
        symbol = ''.join(symbol)
        if symbol in self._namespace:
            if isinstance(self._namespace[symbol], Function('Constant')):
                return self._namespace[symbol]
        function = Function('Tensor')(Symbol(symbol, real=True), *indexing)
        tensor = Tensor(function, self._property['dimension'])
        # reserved keyword for christoffel symbol
        if symbol not in self._namespace and location == 'RHS':
            if symbol[:5] == 'Gamma' and tensor.rank == 3:
                metric = self._property['metric'][symbol[5:-3]] + symbol[5:-3]
                if metric + 'DD' not in self._namespace:
                    raise ParseError('cannot generate christoffel symbol without defined metric \'%s\'' %
                        metric, sentence, position)
                sentence, position = self.lexer.sentence, self.lexer.mark()
                self.parse(self._generate_christoffel(function, self._property['metric']))
                self.lexer.initialize(sentence, position)
                self.lexer.lex()
            elif symbol[:7] == 'epsilon':
                # instantiate permutation (Levi-Civita) symbol using parity
                def sgn(sequence):
                    """ Permutation Signature (Parity)"""
                    cycle_length = 0
                    for n, i in enumerate(sequence[:-1]):
                        for j in sequence[(n + 1):]:
                            if i == j: return 0
                            cycle_length += i > j
                    return (-1)**cycle_length
                index = [chr(105 + n) for n in range(tensor.rank)]
                prefix = '[' * tensor.rank + 'sgn([' + ', '.join(index) + '])'
                suffix = ''.join(' for %s in range(%d)]' % (index[tensor.rank - i], tensor.rank)
                    for i in range(1, tensor.rank + 1))
                tensor.structure = eval(prefix + suffix, {'sgn': sgn})
                tensor.dimension = tensor.rank
                tensor.diff_type = 'symbolic'
                self._define_tensor(tensor)
            else:
                if tensor.rank > 0:
                    if any(suffix in symbol for suffix in ('_d', '_dup', '_cd', '_ld')):
                        raise ParseError('cannot index undefined tensor \'%s\' at position %d' %
                            (symbol, position), sentence, position)
                    i, base_symbol = len(symbol) - 1, symbol
                    while i >= 0:
                        if base_symbol[i] not in ('U', 'D'):
                            base_symbol = base_symbol[:(i + 1)]
                            break
                        i -= 1
                    from itertools import product # TODO U..D.. Ordering
                    for suffix in product(*('UD' if i == 'U' else 'DU' for _, i in Tensor.indexing(function))):
                        symbol_RHS = base_symbol + ''.join(suffix)
                        if symbol_RHS in self._namespace:
                            sentence, position = self.lexer.sentence, self.lexer.mark()
                            diacritic = 'bar'   if 'bar'   in symbol \
                                   else 'hat'   if 'hat'   in symbol \
                                   else 'tilde' if 'tilde' in symbol \
                                   else ''
                            metric = self._namespace[symbol_RHS].metric if self._namespace[symbol_RHS].metric else \
                                self._property['metric'][diacritic] + diacritic
                            indexing_LHS = indexing_RHS = [str(index) for index in indexing]
                            alphabet = (chr(97 + n) for n in range(26))
                            for i, index in enumerate(indexing_LHS):
                                if index in indexing_LHS[:i]:
                                    indexing_LHS[i] = next(x for x in alphabet if x not in indexing_LHS)
                            function_LHS = Function('Tensor')(function.args[0],
                                *(Symbol(i) for i in indexing_LHS))
                            latex = Tensor.latex_format(function_LHS) + ' = '
                            for i, (idx, pos) in enumerate(Tensor.indexing(function_LHS)):
                                if pos != suffix[i]:
                                    indexing_RHS[i] = next(x for x in alphabet if x not in indexing_LHS)
                                    if len(str(idx)) > 1: idx = '\\' + str(idx)
                                    if pos == 'U':
                                        latex += '\\text{%s}^{%s %s} ' % (metric, idx, indexing_RHS[i])
                                    else:
                                        latex += '\\text{%s}_{%s %s} ' % (metric, idx, indexing_RHS[i])
                            latex += Tensor.latex_format(Function('Tensor')(Symbol(symbol_RHS, real=True), *indexing_RHS))
                            diff_type = self._namespace[symbol_RHS].diff_type
                            if diff_type or self._namespace[symbol_RHS].metric:
                                latex += ' % assign '
                                if diff_type:
                                    latex += '-diff_type=' + diff_type + ' '
                                if self._namespace[symbol_RHS].metric:
                                    latex += '-metric=\'' + metric + '\' '
                                latex += '\'' + symbol + '\''
                            self.parse(latex)
                            self.lexer.initialize(sentence, position)
                            self.lexer.lex()
                            return function
                    raise ParseError('cannot index undefined tensor \'%s\' at position %d' %
                        (symbol, position), sentence, position)
                else: self._define_tensor(tensor)
        return function

    # <SYMBOL> -> <LETTER> | <DIACRITIC> '{' <SYMBOL> '}' | <TEXT_CMD> '{' <LETTER> { '_' | <LETTER> | <INTEGER> }* '}'
    def _symbol(self):
        lexeme = self.lexer.lexeme
        if self.accept('LETTER'):
            return lexeme
        if self.accept('DIACRITIC'):
            self.expect('LBRACE')
            symbol = self._symbol() + lexeme[1:]
            self.expect('RBRACE')
            return symbol
        if self.accept('TEXT_CMD'):
            self.expect('LBRACE')
            symbol = [self.lexer.lexeme]
            self.expect('LETTER')
            while any(self.peek(token) for token in
                    ('UNDERSCORE', 'LETTER', 'INTEGER')):
                symbol.append(self.lexer.lexeme)
                self.lexer.lex()
            self.expect('RBRACE')
            return ''.join(symbol).replace('\\', '')
        sentence, position = self.lexer.sentence, self.lexer.mark()
        raise ParseError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <LOWER_INDEX> -> <LETTER> | <INTEGER> | '{' { <LETTER> [ '_' ( <LETTER> | <INTEGER> | '{' { <LETTER> | <INTEGER> }+ '}' ) ] | <INTEGER> }*
    #   [ ( ',' | ';' ) { <LETTER> [ '_' ( <LETTER> | <INTEGER> | '{' { <LETTER> | <INTEGER> }+ '}' ) ] }+ ] '}'
    def _lower_index(self):
        indexing, covariant = [], False
        def append_index():
            index = self._strip(self.lexer.lexeme)
            if self.peek('LETTER'):
                index = Symbol(index, real=True)
                if index in self._property['basis']:
                    index = Integer(self._property['basis'].index(index))
            else: index = Integer(index)
            self.lexer.lex()
            indexing.append(index)
        order = 0
        if self.peek('LETTER') or self.peek('INTEGER'):
            append_index()
            return indexing, order, covariant
        if self.accept('LBRACE'):
            while self.peek('LETTER') or self.peek('INTEGER'):
                append_index()
                if self.accept('UNDERSCORE'):
                    grouped = self.accept('LBRACE')
                    if not any(self.peek(token) for token in ('LETTER', 'INTEGER')):
                        sentence, position = self.lexer.sentence, self.lexer.mark()
                        raise ParseError('unexpected \'%s\' at position %d' %
                            (sentence[position], position), sentence, position)
                    index = self._strip(self.lexer.lexeme)
                    self.lexer.lex()
                    index = str(indexing[-1]) + '_' + index
                    indexing[-1] = Symbol(index, real=True)
                    if grouped:
                        while self.peek('LETTER') or self.peek('INTEGER'):
                            index = self._strip(self.lexer.lexeme)
                            self.lexer.lex()
                            index = str(indexing[-1]) + index
                            indexing[-1] = Symbol(index, real=True)
                        self.expect('RBRACE')
            if self.accept('COMMA'):
                while self.peek('LETTER'):
                    order += 1
                    append_index()
            elif self.accept('SEMICOLON'):
                covariant = True
                while self.peek('LETTER'):
                    order += 1
                    append_index()
            self.expect('RBRACE')
            return indexing, order, covariant
        sentence, position = self.lexer.sentence, self.lexer.mark()
        raise ParseError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <UPPER_INDEX> -> <LETTER> | <INTEGER> | '{' { <LETTER> | <INTEGER> }* '}'
    def _upper_index(self):
        indexing = []
        def append_index():
            index = self._strip(self.lexer.lexeme)
            if self.peek('LETTER'):
                index = Symbol(index, real=True)
                if index in self._property['basis']:
                    index = Integer(self._property['basis'].index(index))
            else: index = Integer(index)
            self.lexer.lex()
            indexing.append(index)
        if self.peek('LETTER') or self.peek('INTEGER'):
            append_index()
            return indexing
        if self.accept('LBRACE'):
            while self.peek('LETTER') or self.peek('INTEGER'):
                append_index()
                if self.accept('UNDERSCORE'):
                    grouped = self.accept('LBRACE')
                    index = self._strip(self.lexer.lexeme)
                    self.expect('INTEGER')
                    index = str(indexing[-1]) + '_' + index
                    indexing[-1] = Symbol(index, real=True)
                    if grouped: self.expect('RBRACE')
            self.expect('RBRACE')
            return indexing
        sentence, position = self.lexer.sentence, self.lexer.mark()
        raise ParseError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <NUMBER> -> <RATIONAL> | <DECIMAL> | <INTEGER> | <PI>
    def _number(self):
        number = self.lexer.lexeme
        if self.accept('RATIONAL'):
            rational = re.match(r'(\-?[0-9]+)\/(\-?[1-9][0-9]*)', number)
            return Rational(rational.group(1), rational.group(2))
        if self.accept('DECIMAL'):
            return Float(number)
        if self.accept('INTEGER'):
            return Integer(number)
        if self.accept('PI'):
            return pi
        sentence, position = self.lexer.sentence, self.lexer.mark()
        raise ParseError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    def _define_tensor(self, tensor, zero=False):
        symbol, dimension = tensor.symbol, tensor.dimension
        if not tensor.structure:
            tensor.structure = Symbol(symbol, real=True) if tensor.rank == 0 \
                else symdef(tensor.rank, symbol if not zero else None, tensor.symmetry, dimension)
        if symbol in self._namespace:
            # pylint: disable=unused-argument
            def formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
                return '%s: %s\n' % (category.__name__, message)
            warnings.formatwarning = formatwarning
            # throw warning whenever duplicate namespace variable
            warnings.warn(symbol, OverrideWarning)
        self._namespace[symbol] = tensor

    def _define_pardrv(self, function, location, diff_type, index):
        if not diff_type or diff_type == 'symbolic':
            return Derivative(function, index)
        symbol, indices = str(function.args[0]), list(function.args[1:]) + [index]
        suffix = '_d'   if diff_type == 'dD' \
            else '_dup' if diff_type == 'dupD' \
            else ''
        tensor = self._namespace[symbol]
        symbol = symbol + ('' if suffix in symbol else suffix) + 'D'
        sentence, position = self.lexer.sentence, self.lexer.mark()
        if symbol not in self._namespace:
            if location == 'RHS' and tensor.equation:
                LHS, RHS = tensor.equation[0]
                tree, idx_set = ExprTree(tensor.equation[1]), set()
                for subtree in tree.preorder():
                    subexpr = subtree.expr
                    if subexpr.func == Function('Tensor'):
                        idx_set.update(subexpr.args[1:])
                idx_set = {str(index) for index in idx_set}
                if str(index) in idx_set:
                    alphabet = (chr(97 + n) for n in range(26))
                    index = next(x for x in alphabet if x not in idx_set)
                operator = '\\partial_' + ('\\' if len(str(index)) > 1 else '') + str(index)
                self.parse('%s %s = %s (%s)' % (operator, LHS, operator, RHS))
                self.lexer.initialize(sentence, position)
                self.lexer.lex()
        function = Function('Tensor')(Symbol(symbol, real=True), *indices)
        if symbol not in self._namespace:
            symmetry = 'nosym'
            if len(symbol.split(suffix)[1]) == 2:
                position = len(indices) - 2
                symmetry = 'sym%d%d' % (position, position + 1)
            if tensor.symmetry and tensor.symmetry != 'nosym':
                symmetry = tensor.symmetry + ('_' + symmetry if symmetry != 'nosym' else '')
            self._define_tensor(Tensor(function, tensor.dimension,
                symmetry=symmetry, diff_type=tensor.diff_type))
        return function

    def _expand_product(self, location, suffix_1, suffix_2, index=None):
        func_list, product = [], None
        if any(self.peek(i) for i in ('LPAREN', 'LBRACK', 'LBRACE')):
            subexpr = self._subexpr()
            tree = ExprTree(subexpr)
            # insert temporary symbol '_x' for symbolic differentiation
            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func in (Function('Tensor'), Derivative):
                    subtree.expr = Function('Function')(subexpr, Symbol('_x'))
                    del subtree.children[:]
            expr = tree.reconstruct()
            # differentiate the expression, including product rule expansion
            tree = ExprTree(diff(expr, Symbol('_x')))
            # remove temporary symbol '_x' from tensor function
            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func == Derivative:
                    function = subexpr.args[0].args[0]
                    if function.func == Derivative:
                        base_func = function.args[0]
                    else: base_func = function
                    symbol, indices = str(base_func.args[0]), list(base_func.args[1:])
                    if function.func == Derivative:
                        symbol += '_dD'
                        for pardrv_index, _ in function.args[1:]:
                            indices.append(pardrv_index)
                        function = Function('Tensor')(Symbol(symbol, real=True), *indices)
                    if index: indices.append(index)
                    name_list = re.split(r'_(cd|ld)', symbol)
                    if len(name_list) > 1:
                        if name_list[-2] == 'cd' != suffix_1[:2] or \
                           name_list[-2] == 'ld' != suffix_1[:2]:
                            symbol += '_' + suffix_1
                    else: symbol += '_' + suffix_1
                    symbol += suffix_2
                    subtree.expr = Function('Tensor')(Symbol(symbol, real=True), *indices)
                    func_list.append((symbol, function))
                    del subtree.children[:]
                elif subexpr.func == Function('Function'):
                    subtree.expr = subexpr.args[0]
                    del subtree.children[:]
            product = tree.reconstruct()
        else:
            function = self._operator(location)
            if function.func == Derivative:
                base_func = function.args[0]
            else: base_func = function
            symbol, indices = str(base_func.args[0]), list(base_func.args[1:])
            if function.func == Derivative:
                symbol += '_dD'
                for pardrv_index, _ in function.args[1:]:
                    indices.append(pardrv_index)
                function = Function('Tensor')(Symbol(symbol, real=True), *indices)
            if index: indices.append(index)
            name_list = re.split(r'_(cd|ld)', symbol)
            if len(name_list) > 1:
                if name_list[-2] == 'cd' != suffix_1[:2] or \
                   name_list[-2] == 'ld' != suffix_1[:2]:
                    symbol += '_' + suffix_1
            else: symbol += '_' + suffix_1
            symbol += suffix_2
            func_list.append((symbol, function))
            product = Function('Tensor')(Symbol(symbol, real=True), *indices)
        return func_list, product

    @staticmethod
    def _separate_indexing(indexing, symbol_LHS):
        free_index, bound_index = [], []
        indexing = [(str(idx), pos) for idx, pos in indexing]
        # iterate over every unique index in the subexpression
        for index in uniquify([idx for idx, _ in indexing]):
            count = U = D = 0; index_tuple = []
            # count index occurrence and position occurrence
            for index_, position in indexing:
                if index_ == index:
                    index_tuple.append((index_, position))
                    if position == 'U': U += 1
                    if position == 'D': D += 1
                    count += 1
            # identify every bound index on the RHS
            if count > 1:
                if count != 2 or U != D:
                    # raise exception upon violation of the following rule:
                    # a bound index must appear exactly once as a superscript
                    # and exactly once as a subscript in any single term
                    raise TensorError('illegal bound index \'%s\' in %s' % (index, symbol_LHS))
                bound_index.append(index)
            # identify every free index on the RHS
            else: free_index.extend(index_tuple)
        return uniquify(free_index), bound_index

    def _summation(self, LHS, RHS):
        rank, indexing = Tensor(LHS).rank, []
        tree = ExprTree(LHS)
        for subtree in tree.preorder():
            subexpr = subtree.expr
            if subexpr.func == Function('Tensor'):
                for index, position in Tensor.indexing(subexpr):
                    if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                        indexing.append((index, position))
            elif subexpr.func == Derivative:
                for index, _ in subexpr.args[1:]:
                    if index not in self._property['basis']:
                        if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                            indexing.append((index, 'D'))
        symbol_LHS = Tensor(LHS).symbol
        # construct a tuple list of every LHS free index
        free_index_LHS, _ = self._separate_indexing(indexing, symbol_LHS)
        # construct a tuple list of every RHS free index
        free_index_RHS = []
        iterable = RHS.args if RHS.func == Add else [RHS]
        LHS, RHS = Tensor(LHS).array_format(LHS), srepr(RHS)
        for element in iterable:
            original, idx_map = srepr(element), {}
            if original[0] == '-':
                original = original[1:]
            modified = original
            indexing = []
            tree = ExprTree(element)
            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func == Function('Tensor'):
                    symbol = str(subexpr.args[0])
                    dimension = self._namespace[symbol].dimension
                    for index in subexpr.args[1:]:
                        upper_bound = dimension
                        if str(index) in self._property['index']:
                            upper_bound = self._property['index'][str(index)]
                        if str(index) in idx_map and upper_bound != idx_map[str(index)]:
                            raise ParseError('inconsistent dimension for index \'%s\'' %
                                index, self.lexer.sentence)
                        idx_map[str(index)] = upper_bound
                    function = Tensor(subexpr).array_format(subexpr)
                    modified = modified.replace(srepr(subexpr), function)
                    for index, position in Tensor.indexing(subexpr):
                        if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                            indexing.append((index, position))
                elif subexpr.func == Function('Constant'):
                    constant = str(subexpr.args[0])
                    modified = modified.replace(srepr(subexpr), constant)
                elif subexpr.func == Derivative:
                    argument = subexpr.args[0]
                    derivative = 'diff(' + srepr(argument)
                    symbol = str(argument.args[0])
                    dimension = self._namespace[symbol].dimension
                    for index, order in subexpr.args[1:]:
                        upper_bound = dimension
                        if str(index) in self._property['index']:
                            upper_bound = self._property['index'][str(index)]
                        if str(index) in idx_map and upper_bound != idx_map[str(index)]:
                            raise ParseError('inconsistent dimension for index \'%s\'' %
                                index, self.lexer.sentence)
                        idx_map[str(index)] = upper_bound
                        if index not in self._property['basis']:
                            if not self._property['basis']:
                                message = 'cannot differentiate symbolically without specifying a basis'
                                raise ParseError(message, self.lexer.sentence)
                            derivative += ', (basis[%s], %s)' % (index, order)
                            if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                                indexing.append((index, 'D'))
                        else: derivative += ', (%s, %s)' % (index, order)
                    derivative += ')'
                    modified = modified.replace(srepr(subexpr), derivative)
                    tmp = srepr(subexpr).replace(srepr(argument), Tensor(argument).array_format(argument))
                    modified = modified.replace(tmp, derivative)
            free_index, bound_index = self._separate_indexing(indexing, symbol_LHS)
            free_index_RHS.append(free_index)
            # generate implied summation over every bound index
            for idx in bound_index:
                modified = 'sum(%s for %s in range(%d))' % (modified, idx, idx_map[idx])
            RHS = RHS.replace(original, modified)
        for i in range(len(free_index_RHS)):
            if sorted(free_index_LHS) != sorted(free_index_RHS[i]):
                # raise exception upon violation of the following rule:
                # a free index must appear in every term with the same
                # position and cannot be summed over in any term
                set_LHS = set(idx for idx, _ in free_index_LHS)
                set_RHS = set(idx for idx, _ in free_index_RHS[i])
                raise TensorError('unbalanced free index %s in %s' % \
                    (set_LHS.symmetric_difference(set_RHS), symbol_LHS))
        # generate tensor instantiation with implied summation
        if symbol_LHS in self._namespace:
            equation = len(free_index_LHS) * '    ' + '%s = %s' % (LHS, RHS)
            for i, (idx, _) in enumerate(reversed(free_index_LHS)):
                indent = len(free_index_LHS) - (i + 1)
                equation = indent * '    ' + 'for %s in range(%d):\n' % (idx, idx_map[idx]) + equation
            equation = [equation]
        else:
            for idx, _ in reversed(free_index_LHS):
                RHS = '[%s for %s in range(%d)]' % (RHS, idx, idx_map[idx])
            equation = [LHS.split('[')[0], RHS]
        if free_index_LHS:
            dimension = idx_map[free_index[0][0]]
            if any(idx_map[index] != dimension for index, _ in free_index):
                raise ParseError('inconsistent free index dimension', self.lexer.sentence)
            LHS_dimension = dimension
        else: LHS_dimension = self._property['dimension']
        # shift tensor indexing forward whenever dimension > upper bound
        for subtree in tree.preorder():
            subexpr = subtree.expr
            if subexpr.func == Function('Tensor'):
                symbol = str(subexpr.args[0])
                dimension = self._namespace[symbol].dimension
                tensor = Tensor(subexpr, dimension)
                indexing = Tensor.indexing(subexpr)
                for index in subexpr.args[1:]:
                    if str(index) in self._property['index']:
                        upper_bound = self._property['index'][str(index)]
                        if dimension > upper_bound:
                            shift = dimension - upper_bound
                            for i, (idx, pos) in enumerate(indexing):
                                if str(idx) == str(index):
                                    indexing[i] = ('%s + %s' % (idx, shift), pos)
                equation[-1] = equation[-1].replace(tensor.array_format(subexpr), tensor.array_format(indexing))
        return ' = '.join(equation), LHS_dimension

    def _generate_metric(self, symbol, dimension, diff_type):
        latex_config = ''
        if 'U' in symbol:
            prefix = r'\epsilon_{' + ' '.join('i_' + str(i) for i in range(1, 1 + dimension)) + '} ' + \
                     r'\epsilon_{' + ' '.join('j_' + str(i) for i in range(1, 1 + dimension)) + '} '
            det_latex = prefix + ' '.join(r'\text{{{symbol}}}^{{i_{n} j_{n}}}'.format(symbol=symbol[:-2], n=i) for i in range(1, 1 + dimension))
            inv_latex = prefix + ' '.join(r'\text{{{symbol}}}^{{i_{n} j_{n}}}'.format(symbol=symbol[:-2], n=i) for i in range(2, 1 + dimension))
            latex_config += r"""
                \text{{{symbol}det}} = \frac{{1}}{{({dimension})({factorial})}} {det_latex} \\
                \text{{{symbol}}}_{{i_1 j_1}} = \frac{{1}}{{{factorial}}} \text{{{symbol}det}}^{{{{-1}}}} ({inv_latex})
            """.format(symbol=symbol[:-2], inv_symbol=symbol.replace('U', 'D'), dimension=dimension,
                    factorial=math.factorial(dimension - 1), det_latex=det_latex, inv_latex=inv_latex)
            if diff_type:
                latex_config += '\n' + r"% assign -diff_type={diff_type} '{symbol}det', '{inv_symbol}'" \
                    .format(diff_type=diff_type, symbol=symbol[:-2], inv_symbol=symbol.replace('U', 'D'))
        else:
            prefix = r'\epsilon^{' + ' '.join('i_' + str(i) for i in range(1, 1 + dimension)) + '} ' + \
                     r'\epsilon^{' + ' '.join('j_' + str(i) for i in range(1, 1 + dimension)) + '} '
            det_latex = prefix + ' '.join(r'\text{{{symbol}}}_{{i_{n} j_{n}}}'.format(symbol=symbol[:-2], n=i) for i in range(1, 1 + dimension))
            inv_latex = prefix + ' '.join(r'\text{{{symbol}}}_{{i_{n} j_{n}}}'.format(symbol=symbol[:-2], n=i) for i in range(2, 1 + dimension))
            latex_config += r"""
                \text{{{symbol}det}} = \frac{{1}}{{({dimension})({factorial})}} {det_latex} \\
                \text{{{symbol}}}^{{i_1 j_1}} = \frac{{1}}{{{factorial}}} \text{{{symbol}det}}^{{{{-1}}}} ({inv_latex})
            """.format(symbol=symbol[:-2], inv_symbol=symbol.replace('D', 'U'), dimension=dimension,
                    factorial=math.factorial(dimension - 1), det_latex=det_latex, inv_latex=inv_latex)
            if diff_type:
                latex_config += '\n' + r"% assign -diff_type={diff_type} '{symbol}det', '{inv_symbol}'" \
                    .format(diff_type=diff_type, symbol=symbol[:-2], inv_symbol=symbol.replace('D', 'U'))
        return latex_config

    @staticmethod
    def _generate_christoffel(function, metric):
        symbol = str(function.args[0])[:-3]
        indexing = [str(index) for index in function.args[1:]]
        alphabet = (chr(97 + n) for n in range(26))
        for i, index in enumerate(indexing):
            if index in indexing[:i]:
                indexing[i] = next(x for x in alphabet if x not in indexing)
        diacritic = 'bar'   if 'bar'   in symbol \
               else 'hat'   if 'hat'   in symbol \
               else 'tilde' if 'tilde' in symbol \
               else ''
        prefix = '\\' if len(metric[diacritic]) > 1 else ''
        metric = '\\%s{%s}' % (diacritic, prefix + metric[diacritic]) if diacritic \
            else prefix + metric[diacritic]
        if diacritic: symbol = '\\%s{\\text{%s}}' % (diacritic, symbol[:-len(diacritic)])
        else: symbol = '\\text{' + symbol + '}'
        indexing = [('\\' if len(str(index)) > 1 else '') + str(index) for index in indexing]
        bound_index = next(x for x in (chr(97 + n) for n in range(26)) if x not in indexing)
        return (('{symbol}^{i1}_{{{i2} {i3}}} = \\frac{{1}}{{2}} {metric}^{{{i1} {bound_index}}} (\\partial_{i2} {metric}_{{{i3} {bound_index}}} + \\partial_{i3} {metric}_{{{bound_index} {i2}}} - \\partial_{bound_index} {metric}_{{{i2} {i3}}})')
                .format(i1 = indexing[0], i2 = indexing[1], i3 = indexing[2], symbol = symbol, metric = metric, bound_index = bound_index))

    @staticmethod
    def _generate_covdrv(function, covdrv_index, symbol=None, diacritic=None):
        indexing = [str(index) for index in function.args[1:]] + [str(covdrv_index)]
        alphabet = (chr(97 + n) for n in range(26))
        for i, index in enumerate(indexing):
            if index in indexing[:i]:
                indexing[i] = next(x for x in alphabet if x not in indexing)
        covdrv_index = indexing[-1]
        if len(str(covdrv_index)) > 1:
            covdrv_index = '\\' + str(covdrv_index)
        latex = Tensor.latex_format(Function('Tensor')(function.args[0],
            *(Symbol(i) for i in indexing[:-1])))
        LHS = ('\\%s{\\nabla}' % diacritic if diacritic else '\\nabla') + ('_%s %s' % (covdrv_index, latex))
        RHS = '\\partial_%s %s' % (covdrv_index, latex)
        for index, (_, position) in zip(indexing, Tensor.indexing(function)):
            alphabet = (chr(97 + n) for n in range(26))
            bound_index = next(x for x in alphabet if x not in indexing)
            latex = Tensor.latex_format(Function('Tensor')(function.args[0],
                *(Symbol(bound_index) if i == index else Symbol(i) for i in indexing[:-1])))
            if len(index) > 1:
                index = '\\' + index
            RHS += ' + ' if position == 'U' else ' - '
            RHS += '\\%s{\\text{Gamma}}' % diacritic if diacritic else '\\text{Gamma}'
            if position == 'U':
                RHS += '^%s_{%s %s} (%s)' % (index, bound_index, covdrv_index, latex)
            else:
                RHS += '^%s_{%s %s} (%s)' % (bound_index, index, covdrv_index, latex)
        config = (' % assign -diff_type=dD \'' + symbol + '\'') if symbol else ''
        return LHS + ' = ' + RHS + config

    @staticmethod
    def _generate_liedrv(function, vector, weight=None):
        if len(str(vector)) > 1:
            vector = '\\text{' + str(vector) + '}'
        indexing = [str(index) for index, _ in Tensor.indexing(function)]
        alphabet = (chr(97 + n) for n in range(26))
        for i, index in enumerate(indexing):
            if index in indexing[:i]:
                indexing[i] = next(x for x in alphabet if x not in indexing)
        latex = Tensor.latex_format(function)
        LHS = '\\mathcal{L}_%s %s' % (vector, latex)
        bound_index = next(x for x in alphabet if x not in indexing)
        RHS = '%s^%s \\partial_%s %s' % (vector, bound_index, bound_index, latex)
        for index, position in Tensor.indexing(function):
            latex = Tensor.latex_format(Function('Tensor')(function.args[0],
                *(Symbol(bound_index) if i == str(index) else Symbol(i) for i in indexing)))
            if len(str(index)) > 1:
                index = '\\' + str(index)
            if position == 'U':
                RHS += ' - (\\partial_%s %s^%s) %s' % (bound_index, vector, index, latex)
            else:
                RHS += ' + (\\partial_%s %s^%s) %s' % (index, vector, bound_index, latex)
        if weight:
            latex = Tensor.latex_format(function)
            RHS += ' + (%s)(\\partial_%s %s^%s) %s' % (weight, bound_index, vector, bound_index, latex)
        return LHS + ' = ' + RHS

    @staticmethod
    def _strip(symbol):
        return symbol[1:] if symbol[0] == '\\' else symbol

    def peek(self, token):
        return self.lexer.token == token

    def accept(self, token):
        if self.peek(token):
            self.lexer.lex()
            return True
        return False

    def expect(self, token):
        if not self.accept(token):
            sentence, position = self.lexer.sentence, self.lexer.mark()
            raise ParseError('expected token %s at position %d' %
                (token, position), sentence, position)

class ParseError(Exception):
    """ Invalid LaTeX Sentence """

    def __init__(self, message, sentence=None, position=None):
        if position is not None:
            length = 0
            for i, substring in enumerate(sentence.split('\n')):
                if position - length <= len(substring):
                    sentence = substring.lstrip()
                    position += len(sentence) - len(substring) - length
                    break
                length += len(substring) + 1
            super(ParseError, self).__init__('%s\n%s^\n' % (sentence, (12 + position) * ' ') + message)
        else: super(ParseError, self).__init__(message)

class Tensor:
    """ Tensor Structure """

    def __init__(self, function, dimension=None, structure=None, equation=None,
            symmetry=None, diff_type=None, metric=None, weight=None):
        self.symbol      = str(function.args[0])
        self.rank        = 0
        for symbol in re.split(r'_d|_dup|_cd|_ld', self.symbol):
            for character in reversed(symbol):
                if character in ('U', 'D'):
                    self.rank += 1
                else: break
        self.dimension   = dimension
        self.structure   = structure
        self.equation    = equation
        self.symmetry    = symmetry
        self.diff_type   = diff_type
        self.metric      = metric
        self.weight      = weight

    @staticmethod
    def indexing(function):
        """ Tensor Indexing from SymPy Function """
        symbol, indices = function.args[0], function.args[1:]
        i, indexing = len(indices) - 1, []
        for symbol in reversed(re.split(r'_d|_dup|_cd|_ld', str(symbol))):
            for character in reversed(symbol):
                if character in ('U', 'D'):
                    indexing.append((indices[i], character))
                else: break
                i -= 1
        return list(reversed(indexing))

    # TODO change method type to static (class) method
    def array_format(self, function):
        """ Tensor Notation for Array Formatting """
        if isinstance(function, Function('Tensor')):
            indexing = self.indexing(function)
        else: indexing = function
        if not indexing:
            return self.symbol
        return self.symbol + ''.join(['[' + str(index) + ']' for index, _ in indexing])

    @staticmethod
    def latex_format(function):
        """ Tensor Notation for LaTeX Formatting """
        symbol, indexing = str(function.args[0]), Tensor.indexing(function)
        operator, i_2 = '', len(symbol)
        for i_1 in range(len(symbol), 0, -1):
            subsym = symbol[i_1:i_2]
            if '_d' in subsym:
                suffix = re.split('_d|_dup', subsym)[-1]
                for _ in reversed(suffix):
                    index = str(indexing.pop()[0])
                    if len(index) > 1:
                        index = '\\' + index
                    operator += '\\partial_' + index + ' '
                i_2 = i_1
            elif '_cd' in subsym:
                suffix = subsym.split('_cd')[-1]
                diacritic = 'bar'   if 'bar'   in suffix \
                       else 'hat'   if 'hat'   in suffix \
                       else 'tilde' if 'tilde' in suffix \
                       else None
                if diacritic:
                    suffix = suffix[len(diacritic):]
                for position in reversed(suffix):
                    index = str(indexing.pop()[0])
                    if '_' in index:
                        index, subscript = index.split('_')
                        if len(index) > 1:
                            index = '\\' + index
                        index += '_{' + subscript + '}'
                    else:
                        if len(index) > 1:
                            index = '\\' + index
                    operator += '\\' + diacritic + '{\\nabla}' if diacritic \
                        else '\\nabla'
                    if position == 'U':
                        operator += '^' + index
                    else:
                        operator += '_' + index
                    operator += ' '
                i_2 = i_1
            elif '_ld' in subsym:
                vector = re.split('_ld', subsym)[-1]
                if len(vector) > 1:
                    vector = '\\text{' + vector + '}'
                operator += '\\mathcal{L}_' + vector + ' '
                i_2 = i_1
        diacritic = 'bar'   if 'bar'   in symbol[i_1:i_2] \
               else 'hat'   if 'hat'   in symbol[i_1:i_2] \
               else 'tilde' if 'tilde' in symbol[i_1:i_2] \
               else ''
        symbol = re.split(r'_d|_dup|_cd|_ld', symbol)[0]
        for i, character in enumerate(reversed(symbol)):
            if character not in ('U', 'D'):
                symbol = symbol[:len(symbol) - i]; break
        latex = [symbol, [], []]
        if len(latex[0]) > 1:
            latex[0] = '\\text{' + str(latex[0]) + '}'
        latex[0] = operator + latex[0]
        U_count, D_count = 0, 0
        for index, position in indexing:
            index = str(index)
            if '_' in index:
                index, subscript = index.split('_')
                if len(index) > 1:
                    index = '\\' + index
                index += '_{' + subscript + '}'
            else:
                if len(index) > 1:
                    index = '\\' + index
            if position == 'U':
                latex[1].append(index)
                U_count += 1
            else:
                latex[2].append(index)
                D_count += 1
        latex[1] = ' '.join(latex[1])
        latex[2] = ' '.join(latex[2])
        if U_count > 0:
            if U_count > 1:
                latex[1] = '^{' + latex[1] + '}'
            else: latex[1] = '^' + latex[1]
        if D_count > 0:
            if D_count > 1:
                latex[2] = '_{' + latex[2] + '}'
            else: latex[2] = '_' + latex[2]
        return ''.join(latex)

    def __repr__(self):
        if self.rank == 0:
            return 'Scalar(%s)' % self.symbol
        return 'Tensor(%s, %dD)' % (self.symbol, self.dimension)

    __str__ = __repr__

def ignore_override(option):
    """ Ignore Override Warning

        :arg: boolean value
    """
    action = 'ignore' if option else 'default'
    warnings.filterwarnings(action, category=OverrideWarning)

def delete_namespace(**kwargs):
    """ Delete Namespace (Package Global Scope)

        :arg: automatic mode [default: disabled]
    """
    if 'automatic' in kwargs:
        automatic = kwargs['automatic']
        Parser.continue_parsing = not automatic
    Parser._namespace, Parser._property = {}, {}

# pylint: disable = unused-argument
inside_ipython = False
try:
    get_ipython()
    inside_ipython = True
except NameError: pass
if inside_ipython:
    from IPython.core.magic import register_cell_magic
    @register_cell_magic
    def parse_latex(line, cell):
        try: return parse(cell, ipython=True)
        except (ParseError, TensorError) as Error:
            ErrorName = 'ParseError' \
                if type(Error).__name__ == 'ParseError' else 'TensorError'
            print(ErrorName + ': ' + str(Error))

class ParseOutput(tuple):
    """ Output Structure for IPython (Jupyter) """

    # pylint: disable = super-init-not-called
    def __init__(self, iterable, sentence):
        self.iterable = iterable
        self.sentence = sentence

    # pylint: disable = unused-argument
    def __new__(cls, iterable, sentence):
        return super(ParseOutput, cls).__new__(cls, iterable)

    def _repr_latex_(self):
        return r'\[' + self.sentence + r'\]'

class TensorError(Exception):
    """ Invalid Tensor Indexing or Dimension """
class OverrideWarning(UserWarning):
    """ Overridden Namespace Variable """

def parse_expr(sentence, verbose=False):
    """ Convert LaTeX Sentence to SymPy Expression (Expression Mode)

        :arg: latex sentence (raw string)
        :arg: verbose mode [default: disabled]
        :return: expression
    """
    return Parser(verbose).parse(sentence, expression=True)

def parse(sentence, verbose=False, ipython=False):
    """ Convert LaTeX Sentence to SymPy Expression

        :arg: latex sentence (raw string)
        :arg: verbose mode [default: disabled]
        :arg: ipython mode [default: disabled]
        :return: namespace
    """
    if not Parser.continue_parsing:
        delete_namespace()
    _namespace = Parser._namespace.copy()
    namespace = Parser(verbose).parse(sentence)
    key_diff = [key for key in namespace if key not in _namespace]
    # inject updated namespace into the previous stack frame
    frame = currentframe().f_back
    if ipython: frame = frame.f_back.f_back
    for key in namespace:
        if isinstance(namespace[key], Tensor):
            tensor = namespace[key]
            if not tensor.equation and tensor.rank == 0:
                if not verbose and key in key_diff:
                    key_diff.remove(key)
            frame.f_globals[key] = namespace[key].structure
        elif isinstance(namespace[key], Function('Constant')):
            frame.f_globals[key] = namespace[key].args[0]
            if not verbose and key in key_diff:
                key_diff.remove(key)
    iterable = tuple(key_diff) if not verbose \
          else tuple(namespace[key] for key in key_diff)
    return ParseOutput(iterable, sentence)
