import re
import warnings
from typing import Any, Iterator

from ..utils.exceptions import DeprecatedWarning, NRPyLaTeXError


class Scanner:
    sentence: str
    position: int
    eqn_mode: bool
    lexeme: str
    token: str | None
    prev_state: tuple[int, bool]

    def __init__(self) -> None:
        symmetry = r'nosym|(?:sym|anti)[0-9]+(?:_(?:sym|anti)[0-9]+)*'
        alphabet = '|'.join(
            letter
            for letter in (
                r'\\[aA]lpha',
                r'\\[bB]eta',
                r'\\[gG]amma',
                r'\\[dD]elta',
                r'\\[eE]psilon',
                r'\\[zZ]eta',
                r'\\[eE]ta',
                r'\\[tT]heta',
                r'\\[iI]ota',
                r'\\[kK]appa',
                r'\\[lL]ambda',
                r'\\[mM]u',
                r'\\[nN]u',
                r'\\[xX]i',
                r'\\[oO]mikron',
                r'\\[pP]i',
                r'\\[Rr]ho',
                r'\\[sS]igma',
                r'\\[tT]au',
                r'\\[uU]psilon',
                r'\\[pP]hi',
                r'\\[cC]hi',
                r'\\[pP]si',
                r'\\[oO]mega',
                r'\\varepsilon',
                r'\\varkappa',
                r'\\varphi',
                r'\\varpi',
                r'\\varrho',
                r'\\varsigma',
                r'\\vartheta',
                r'[a-zA-Z]',
            )
        )
        self.deprecated: list[tuple[str, str]] = [('\\text', '\\mathrm')]

        token_dict_cfg = [
            ('LINEBREAK', r'\r?\n'),
            ('WHITESPACE', r'\s+'),
            ('COMMENT', r'\%\%([^\n]*|.*$)'),
            ('PERCENT', r'\%'),
            ('STRING', r'\"[^\"]*\"'),
            ('ARROW', r'\-\>'),
            ('DBL_DASH', r'\-\-'),
            ('SYMMETRY', symmetry),
            ('ZEROS_OPT', r'zeros'),
            ('CONST_OPT', r'const'),
            ('DIM_OPT', r'dim'),
            ('SYM_OPT', r'sym'),
            ('WEIGHT_OPT', r'weight'),
            ('SUFFIX_OPT', r'suffix'),
            ('METRIC_OPT', r'metric'),
            ('COORD_KWD', r'coord'),
            ('DEFAULT_KWD', r'default'),
            ('INDEX_KWD', r'index'),
            ('LATIN_KWD', r'latin'),
            ('GREEK_KWD', r'greek'),
            ('IDENTIFIER', rf'{alphabet}(({alphabet})|[_0-9])*'),
            ('INTEGER', r'[0-9]+'),
            ('NEWLINE', r'\\{2}'),
        ]
        self.pattern_cfg: re.Pattern[str] = re.compile(
            '|'.join([f'(?P<{k}>{v})' for k, v in token_dict_cfg])
        )
        self.token_dict_cfg: dict[str, str] = dict(token_dict_cfg)

        token_dict_eqn = [
            ('LINEBREAK', r'\r?\n'),
            ('WHITESPACE', r'\s+'),
            ('COMMENT', r'\%\%([^\n]*|.*$)'),
            ('PERCENT', r'\%'),
            ('RATIONAL', r'[0-9]+\/\-?[1-9][0-9]*'),
            ('DECIMAL', r'[0-9]+\.[0-9]+'),
            ('INTEGER', r'[0-9]+'),
            ('PLUS', r'\+'),
            ('MINUS', r'\-'),
            ('DIVIDE', r'\/'),
            ('EQUAL', r'\='),
            ('CARET', r'\^'),
            ('UNDERSCORE', r'\_'),
            ('APOSTROPHE', r'\''),
            ('COMMA', r'\,'),
            ('COLON', r'\:'),
            ('SEMICOLON', r'\;'),
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('LBRACK', r'\['),
            ('RBRACK', r'\]'),
            ('LBRACE_ESC', r'\\{'),
            ('RBRACE_ESC', r'\\}'),
            ('LBRACE', r'\{'),
            ('RBRACE', r'\}'),
            ('DECLARE_CFG', r'declare'),
            ('REPLACE_CFG', r'replace'),
            ('IGNORE_CFG', r'ignore'),
            ('PAR_SYM', r'\\partial'),
            ('COV_SYM', r'\\nabla'),
            ('LIE_SYM', r'\\mathcal\{L\}'),
            ('EXP_CMD', r'\\exp'),
            ('LOG_CMD', r'\\ln|\\log'),
            ('FRAC_CMD', r'\\frac'),
            ('SQRT_CMD', r'\\sqrt'),
            ('TRIG_CMD', r'\\sinh|\\cosh|\\tanh|\\sin|\\cos|\\tan'),
            ('SUFFIX_KWD', r'suffix'),
            ('NOIMPSUM_KWD', r'noimpsum'),
            ('CONSTANT', r'\\pi'),
            ('DIACRITIC', r'\\hat|\\bar|\\tilde'),
            ('MULTISYMB', rf'\\mathrm{{({alphabet})(({alphabet})|[_0-9])*}}'),
            ('GROUP', r'\\[1-9][0-9]*\*?'),
            ('CHARACTER', alphabet),
            ('NEWLINE', r'\\{2}'),
            ('BACKSLASH', r'\\'),
        ]
        self.pattern_eqn: re.Pattern[str] = re.compile(
            '|'.join([f'(?P<{k}>{v})' for k, v in token_dict_eqn])
        )
        self.token_dict_eqn: dict[str, str] = dict(token_dict_eqn)

    def initialize(self, sentence: str, state: tuple[int, bool] | None = None) -> None:
        while True:
            sentence_ = re.sub(
                r'\\mathrm{([^{}]*)\\mathrm{([^{}]+)}([^{}]*)}', r'\\mathrm{\1\2\3}', sentence
            )
            if sentence_ == sentence:
                break
            sentence = sentence_

        for feature, replacement in self.deprecated:
            if feature in sentence:
                warnings.warn(f'{feature} is deprecated.', DeprecatedWarning)
            sentence = sentence.replace(feature, replacement)

        self.sentence = sentence
        if state is None:
            self.position = 0
            self.eqn_mode = True
            self.token = None
            self.lexeme = ''
            self.prev_state = (0, True)
        else:
            self.position, self.eqn_mode = state
            self.lex()

    def tokenize(self) -> Iterator[str]:
        while self.position < len(self.sentence):
            pattern = self.pattern_eqn if self.eqn_mode else self.pattern_cfg
            token = pattern.match(self.sentence, self.position)
            if token is None:
                raise ScannerError(
                    f"unexpected '{self.sentence[self.position]}'", self.sentence, self.position
                )

            self.prev_state = (self.position, self.eqn_mode)

            # The token match object is guaranteed to have a matched string for the named group
            lastgroup = str(token.lastgroup)

            if lastgroup.endswith('CFG'):
                self.eqn_mode = False
            self.position = token.end()
            if lastgroup in ('COMMENT', 'WHITESPACE') or self.eqn_mode and lastgroup == 'LINEBREAK':
                continue

            self.lexeme = token.group()
            if lastgroup == 'LINEBREAK':
                self.eqn_mode = True
            yield lastgroup

    def lex(self) -> str | None:
        try:
            self.token = next(self.tokenize())
        except StopIteration:
            self.token = None
            self.lexeme = ''
            self.prev_state = (self.position, self.eqn_mode)
        return self.token

    def reset(self, state: tuple[int, bool]) -> None:
        if not hasattr(self, 'sentence') or not self.sentence:
            raise RuntimeError('cannot reset uninitialized scanner')
        self.initialize(self.sentence, state)

    def context(self) -> 'Scanner.ScannerContext':
        return self.ScannerContext(self)

    class ScannerContext:
        def __init__(self, scanner: 'Scanner') -> None:
            self.scanner = scanner
            self.init_args = (scanner.sentence, scanner.prev_state)

        def __enter__(self) -> None:
            return

        def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
            self.scanner.initialize(*self.init_args)


class ScannerError(NRPyLaTeXError):
    def __init__(
        self, message: str, sentence: str | None = None, position: int | None = None
    ) -> None:
        super().__init__(message, sentence, position)
