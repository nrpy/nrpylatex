""" NRPyLaTeX: LaTeX Interface to SymPy (CAS) for General Relativity """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

from IPython.core.magic import Magics, magics_class, line_cell_magic

@magics_class
class ParseMagic(Magics):
    """ NRPyLaTeX IPython Magic """

    @line_cell_magic
    def parse_latex(self, line, cell=None):
        match, kwargs = re.match(r'\s*--([^\s]+)\s*', line), []
        while match:
            kwargs.append(match.group(1))
            line = line[match.span()[-1]:]
            match = re.match(r'\s*--([^\s]+)\s*', line)

        debug = False
        for arg in kwargs:
            if arg == 'reset':
                Parser.initialize(reset=True)
            elif arg == 'debug':
                debug = True

        try:
            sentence = line if cell is None else cell
            state = tuple(Parser._namespace.keys())
            namespace = Parser(debug).parse_latex(sentence)
            if not isinstance(namespace, dict):
                return namespace
            if not namespace: return None

            for key in namespace:
                if isinstance(namespace[key], IndexedSymbol):
                    self.shell.user_ns[key] = namespace[key].structure
                elif isinstance(namespace[key], Function('Constant')):
                    self.shell.user_ns[key] = namespace[key].args[0]

            overridden = [key for key in state if key in namespace]
            return ParseOutput((('*' if symbol in overridden else '')
                + str(symbol) for symbol in namespace.keys()), sentence)

        except NRPyLaTeXError as e:
            print(type(e).__name__ + ': ' + str(e))
