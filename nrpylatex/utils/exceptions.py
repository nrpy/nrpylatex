"""NRPyLaTeX Exceptions"""


class NRPyLaTeXError(Exception):
    def __init__(
        self, message: str, sentence: str | None = None, position: int | None = None
    ) -> None:
        if position is not None and sentence is not None:
            length = 0
            for _, substring in enumerate(sentence.split('\n')):
                if position - length <= len(substring):
                    sentence = substring.lstrip()
                    position += len(sentence) - len(substring) - length
                    break
                length += len(substring) + 1

            spacing = ' ' * (position + 2)
            super().__init__(f'{message}\n  {sentence}\n{spacing}^')
        else:
            super().__init__(message)


class NamespaceError(Exception):
    """Illegal Namespace Import"""


class DeprecatedWarning(Warning):
    """Use of Deprecated Feature"""
