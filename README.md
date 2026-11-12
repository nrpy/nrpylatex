# NRPyLaTeX: A LaTeX interface to computer algebra systems for general relativity

[![CI](https://github.com/nrpy/nrpylatex/actions/workflows/pytest.yaml/badge.svg)](https://github.com/nrpy/nrpylatex/actions/workflows/pytest.yaml)
[![PyPI](https://img.shields.io/pypi/v/nrpylatex.svg)](https://pypi.org/project/nrpylatex/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nrpy/nrpylatex.git/HEAD?filepath=nrpylatex-tutorial.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2111.05861-B31B1B)](https://arxiv.org/abs/2111.05861)

**NRPyLaTeX** enables researchers to write tensorial expressions in LaTeX using Einstein notation and parse these expressions directly into SymPy. To resolve the inherent ambiguities of a typesetting language, NRPyLaTeX uses preprocessor directives embedded entirely within LaTeX comments. This allows users to define metrics, dimensions, symmetries, and custom operators without affecting the rendered manuscript.

By integrating with [NRPy](https://github.com/nrpy/nrpy) and its indexed-symbol conventions, NRPyLaTeX provides a direct pipeline from parsed expressions to optimized C/C++ kernels for numerical relativity. It ensures robustness by catching Einstein notation indexing errors and flagging unresolved LaTeX ambiguities. By embedding everything within the manuscript source, it lowers the barrier to entry, minimizes transcription errors, and transforms publications into shareable, reproducible records of computation, all with seamless IPython/Jupyter integration.

## Installation

To install **NRPyLaTeX** using [PyPI](https://pypi.org/project/nrpylatex/), run the following command in the terminal

```bash
pip install nrpylatex
```

## LaTeX Example ([Kretschmann Scalar](https://en.wikipedia.org/wiki/Kretschmann_scalar))

### Python REPL or Script (*.py)

```python
>>> from nrpylatex import parse_latex
>>> parse_latex(r"""
...     % ignore "\begin{align}" "\end{align}"
...     \begin{align}
...         % declare coord t r theta phi
...         % declare G M c --const
...         % declare metric gDD --zeros
...         g_{t t} &= -\left(1 - \frac{2GM}{c^2 r}\right) \\
...         g_{r r} &=  \left(1 - \frac{2GM}{c^2 r}\right)^{-1} \\
...         g_{\theta \theta} &= r^2 \\
...         g_{\phi \phi} &= r^2 \sin^2{\theta} \\
...         R^\alpha{}_{\beta \mu \nu} &= \partial_\mu \Gamma^\alpha_{\beta \nu} - \partial_\nu \Gamma^\alpha_{\beta \mu}
...             + \Gamma^\alpha_{\mu \gamma} \Gamma^\gamma_{\beta \nu} - \Gamma^\alpha_{\nu \sigma} \Gamma^\sigma_{\beta \mu} \\
...         K &= R^{\alpha \beta \mu \nu} R_{\alpha \beta \mu \nu}
...     \end{align}
... """)
ParsedNamespace(G, M, c, gDD, r, theta, gdet, epsilonUUUU, gUU, GammaUDD, RUDDD, RUUUU, RDDDD, K)
>>> from sympy import simplify
>>> print(simplify(K))
48*G**2*M**2/(c**4*r**6)
```

### IPython REPL or Jupyter Notebook

```python
In [1]: %load_ext nrpylatex
In [2]: %%parse_latex
    ...: % ignore "\begin{align}" "\end{align}"
    ...: \begin{align}
    ...:     % declare coord t r theta phi
    ...:     % declare G M c --const
    ...:     % declare metric gDD --zeros
    ...:     g_{t t} &= -\left(1 - \frac{2GM}{c^2 r}\right) \\
    ...:     g_{r r} &=  \left(1 - \frac{2GM}{c^2 r}\right)^{-1} \\
    ...:     g_{\theta \theta} &= r^2 \\
    ...:     g_{\phi \phi} &= r^2 \sin^2{\theta} \\
    ...:     R^\alpha{}_{\beta \mu \nu} &= \partial_\mu \Gamma^\alpha_{\beta \nu} - \partial_\nu \Gamma^\alpha_{\beta \mu}
    ...:         + \Gamma^\alpha_{\mu \gamma} \Gamma^\gamma_{\beta \nu} - \Gamma^\alpha_{\nu \sigma} \Gamma^\sigma_{\beta \mu} \\
    ...:     K &= R^{\alpha \beta \mu \nu} R_{\alpha \beta \mu \nu}
    ...: \end{align}
Out[2]: ParsedNamespace(G, M, c, gDD, r, theta, gdet, epsilonUUUU, gUU, GammaUDD, RUDDD, RUUUU, RDDDD, K)
In [3]: from sympy import simplify
In [4]: print(simplify(K))
Out[4]: 48*G**2*M**2/(c**4*r**6)
```

## &#167; Exporting Output

If you are using Mathematica instead of SymPy, run the following code to convert your output

```python
from sympy import mathematica_code
    
namespace = parse_latex(...)
for var in namespace:
    exec(f'{var} = mathematica_code({var})')
```

If you are using a different CAS, reference the SymPy [documentation](https://docs.sympy.org/latest/modules/printing.html) to find the relevant printing function.
