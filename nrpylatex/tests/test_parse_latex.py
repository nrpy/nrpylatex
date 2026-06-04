from typing import Any

import sympy as sp

from nrpylatex import Generator, Parser, parse_latex


def parse_latex_reset(expr: str) -> Any:
    return parse_latex(expr, reset=True)


def test_expression_arithmetic() -> None:
    expr = r'-(\frac{2}{3} + 2\sqrt[5]{x + 3})'
    assert str(parse_latex_reset(expr)) == '-2*(x + 3)**(1/5) - 2/3'


def test_expression_exponential() -> None:
    expr = r'e^{\ln{x}} - \tanh{xy}'
    assert str(parse_latex_reset(expr)) == 'x - tanh(x*y)'


def test_expression_trigonometric() -> None:
    expr = r'x\cos{\pi} + \sin{\sin^{-1}{y}}'
    assert str(parse_latex_reset(expr)) == '-x + y'


def test_expression_derivative() -> None:
    expr = r'\partial_x (x^2 + 2x)'
    assert str(parse_latex_reset(expr).doit()) == '2*x + 2'


def test_generation_covdrv() -> None:
    function = sp.Function('Tensor')(sp.Symbol('T'))
    assert Generator.generate_covdrv(function, 'beta') == r'\nabla_{\beta} T = \partial_{\beta} T'

    function = sp.Function('Tensor')(sp.Symbol('TUU'), sp.Symbol('mu'), sp.Symbol('nu'))
    assert (
        Generator.generate_covdrv(function, 'beta')
        == r'\nabla_{\beta} T^{\mu \nu} = \partial_{\beta} T^{\mu \nu} + \mathrm{Gamma}^{\mu}_{i_1 \beta} (T^{i_1 \nu}) + \mathrm{Gamma}^{\nu}_{i_1 \beta} (T^{\mu i_1})'
    )

    function = sp.Function('Tensor')(sp.Symbol('TUD'), sp.Symbol('mu'), sp.Symbol('nu'))
    assert (
        Generator.generate_covdrv(function, 'beta')
        == r'\nabla_{\beta} T^{\mu}_{\nu} = \partial_{\beta} T^{\mu}_{\nu} + \mathrm{Gamma}^{\mu}_{i_1 \beta} (T^{i_1}_{\nu}) - \mathrm{Gamma}^{i_1}_{\nu \beta} (T^{\mu}_{i_1})'
    )

    function = sp.Function('Tensor')(sp.Symbol('TDD'), sp.Symbol('mu'), sp.Symbol('nu'))
    assert (
        Generator.generate_covdrv(function, 'beta')
        == r'\nabla_{\beta} T_{\mu \nu} = \partial_{\beta} T_{\mu \nu} - \mathrm{Gamma}^{i_1}_{\mu \beta} (T_{i_1 \nu}) - \mathrm{Gamma}^{i_1}_{\nu \beta} (T_{\mu i_1})'
    )


def test_generation_nested_covdrv() -> None:
    parse_latex_reset(r"""
        % declare metric gDD --dim 4 --suffix dD
        % declare vU --dim 4 --suffix dD
        T^\mu_\nu = \nabla_\nu v^\mu
    """)
    function = sp.Function('Tensor')(sp.Symbol('vU_cdD'), sp.Symbol('mu'), sp.Symbol('nu'))
    assert (
        Generator.generate_covdrv(function, 'beta')
        == r'\nabla_{\beta} \nabla_{\nu} v^{\mu} = \partial_{\beta} \nabla_{\nu} v^{\mu} + \mathrm{Gamma}^{\mu}_{i_1 \beta} (\nabla_{\nu} v^{i_1}) - \mathrm{Gamma}^{i_1}_{\nu \beta} (\nabla_{i_1} v^{\mu})'
    )


def test_generation_liedrv() -> None:
    function = sp.Function('Tensor')(sp.Symbol('g'))
    assert (
        Generator.generate_liedrv(function, 'beta', 2)
        == r'\mathcal{L}_\mathrm{beta} g = \mathrm{beta}^{i_1} \partial_{i_1} g + (2)(\partial_{i_1} \mathrm{beta}^{i_1}) g'
    )

    function = sp.Function('Tensor')(sp.Symbol('gUU'), sp.Symbol('i'), sp.Symbol('j'))
    assert (
        Generator.generate_liedrv(function, 'beta')
        == r'\mathcal{L}_\mathrm{beta} g^{i j} = \mathrm{beta}^{i_1} \partial_{i_1} g^{i j} - (\partial_{i_1} \mathrm{beta}^{i}) g^{i_1 j} - (\partial_{i_1} \mathrm{beta}^{j}) g^{i i_1}'
    )

    function = sp.Function('Tensor')(sp.Symbol('gUD'), sp.Symbol('i'), sp.Symbol('j'))
    assert (
        Generator.generate_liedrv(function, 'beta')
        == r'\mathcal{L}_\mathrm{beta} g^{i}_{j} = \mathrm{beta}^{i_1} \partial_{i_1} g^{i}_{j} - (\partial_{i_1} \mathrm{beta}^{i}) g^{i_1}_{j} + (\partial_{j} \mathrm{beta}^{i_1}) g^{i}_{i_1}'
    )

    function = sp.Function('Tensor')(sp.Symbol('gDD'), sp.Symbol('i'), sp.Symbol('j'))
    assert (
        Generator.generate_liedrv(function, 'beta')
        == r'\mathcal{L}_\mathrm{beta} g_{i j} = \mathrm{beta}^{i_1} \partial_{i_1} g_{i j} + (\partial_{i} \mathrm{beta}^{i_1}) g_{i_1 j} + (\partial_{j} \mathrm{beta}^{i_1}) g_{i i_1}'
    )


def test_replacement_rule() -> None:
    parse_latex(r"""
        % replace "\1'" -> "\mathrm{\1prime}"
        % replace "\1_{\2*}" -> "\mathrm{\1_\2*}"
        % replace "\1_\2"    -> "\mathrm{\1_\2}"
        % replace "\1^{\2*}" -> "\1^{{\2*}}"
        % replace "\1^\2"    -> "\1^{{\2}}"
    """)
    expr = r"x_n^4 + x'_n \exp{x_n y_n^2}"
    assert str(parse_latex(expr)) == 'x_n**4 + xprime_n*exp(x_n*y_n**2)'
    Parser.initialize(reset=True)


def test_recursive_replacement() -> None:
    parse_latex(r"""
        % replace "K" -> "\mathrm{trK}"
        x = K^{{2}} \\
        y = \mathrm{trK} + x
    """)
    expr = r'K^{{2}} + \mathrm{trK}'
    assert str(parse_latex(expr)) == 'trK**2 + trK'
    Parser.initialize(reset=True)


def test_product_rule() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare index latin --dim 2
        % declare vU wU --dim 2 --suffix dD
        T^{ab}_c = \partial_c (v^a w^b)
    """)
    assert set(ns_vars) == {'vU', 'wU', 'vU_dD', 'wU_dD', 'TUUD'}
    assert str(ns_vars.TUUD[0][0][0]) == 'vU0*wU_dD00 + vU_dD00*wU0'


def test_upwind_suffix() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare index latin --dim 2
        % declare vU --dim 2 --suffix dD
        % declare w --const
        T^a_c = % suffix dupD
        \partial_c (v^a w)
    """)
    assert set(ns_vars) == {'w', 'vU', 'vU_dupD', 'TUD'}
    assert str(ns_vars.TUD) == '[[vU_dupD00*w, vU_dupD01*w], [vU_dupD10*w, vU_dupD11*w]]'


def test_inference_covdrv() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare index latin --dim 4
        % declare metric gDD --dim 4 --suffix dD
        % declare vU --dim 4 --suffix dD
        T^{ab} = \nabla^b v^a
    """)
    assert set(ns_vars) == {
        'gUU',
        'gdet',
        'epsilonUUUU',
        'gDD',
        'vU',
        'vU_dD',
        'gDD_dD',
        'GammaUDD',
        'vU_cdD',
        'vU_cdU',
        'TUU',
    }


def test_inference_pardrv() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare coord x y
        % declare index latin --dim 2
        % declare uD --zeros --dim 2
        u_x = x^2 + 2x \\
        u_y = y\sqrt{x}
        % declare wD vD --dim 2 --suffix dD
        v_a = u_a + w_a \\
        T_{ab} = \partial_b v_a
    """)
    assert set(ns_vars) == {'x', 'y', 'uD', 'wD', 'vD', 'vD_dD', 'wD_dD', 'TDD'}
    assert (
        str(ns_vars.TDD)
        == '[[wD_dD00 + 2*x + 2, wD_dD01], [wD_dD10 + y/(2*sqrt(x)), wD_dD11 + sqrt(x)]]'
    )


def test_notation_pardrv() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare index latin --dim 2
        % declare vD uD wD --dim 2 --suffix dD
        T_{abc} = ((v_a + u_a)_{,b} - w_{a,b})_{,c}
    """)
    assert set(ns_vars) == {
        'vD',
        'uD',
        'wD',
        'TDDD',
        'uD_dD',
        'vD_dD',
        'wD_dD',
        'wD_dDD',
        'uD_dDD',
        'vD_dDD',
    }
    assert str(ns_vars.TDDD[0][0][0]) == 'uD_dDD000 + vD_dDD000 - wD_dDD000'


def test_spherical_riemann() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare coord theta phi
        % declare index latin --dim 2
        % declare index greek --dim 2
        % declare metric gDD --zeros --dim 2
        % declare r --const
        % g_{0 0} = r^2 \\
        % g_{1 1} = r^2 \sin^2{\theta} \\
        % ignore "\begin{align*}" "\end{align*}"
        \begin{align*}
            R^\alpha_{\beta\mu\nu} &= \partial_\mu \Gamma^\alpha_{\beta\nu} - \partial_\nu \Gamma^\alpha_{\beta\mu} + \Gamma^\alpha_{\mu\gamma}\Gamma^\gamma_{\beta\nu} - \Gamma^\alpha_{\nu\sigma}\Gamma^\sigma_{\beta\mu} \\
            R_{\alpha\beta\mu\nu} &= g_{\alpha a} R^a_{\beta\mu\nu} \\
            R_{\beta\nu} &= R^\alpha_{\beta\alpha\nu} \\
            R &= g^{\beta\nu} R_{\beta\nu}
        \end{align*}
    """)
    assert str(ns_vars.GammaUDD[0][1][1]) == '-sin(theta)*cos(theta)'
    assert ns_vars.GammaUDD[1][0][1] - ns_vars.GammaUDD[1][1][0] == 0
    assert str(ns_vars.GammaUDD[1][0][1]) == 'cos(theta)/sin(theta)'
    assert (
        ns_vars.RDDDD[0][1][0][1]
        - (-ns_vars.RDDDD[0][1][1][0])
        + (-ns_vars.RDDDD[1][0][0][1])
        - ns_vars.RDDDD[1][0][1][0]
        == 0
    )
    assert str(ns_vars.RDDDD[0][1][0][1]) == 'r**2*sin(theta)**2'
    assert ns_vars.RDD[0][0] == 1
    assert str(ns_vars.RDD[1][1]) == 'sin(theta)**2'
    assert ns_vars.RDD[0][1] - ns_vars.RDD[1][0] == 0
    assert ns_vars.RDD[0][1] == 0
    assert str(sp.simplify(ns_vars.R)) == '2/r**2'


def test_dimension_reduction() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare gDD --dim 4 --sym sym01
        \gamma_{ij} = g_{ij}
    """)
    assert set(ns_vars) == {'gDD', 'gammaDD'}
    assert (
        str(ns_vars.gammaDD)
        == '[[gDD11, gDD12, gDD13], [gDD12, gDD22, gDD23], [gDD13, gDD23, gDD33]]'
    )


def test_spatial_contraction() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare TUU --dim 3
        % declare vD --dim 2
        % declare index i --dim 2
        w^a = T^{a i} v_i
    """)
    assert set(ns_vars) == {'TUU', 'vD', 'wU'}
    assert (
        str(ns_vars.wU) == '[TUU01*vD0 + TUU02*vD1, TUU11*vD0 + TUU12*vD1, TUU21*vD0 + TUU22*vD1]'
    )


def test_inference_indexing() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare metric gDD --dim 3
        % declare ADDD --dim 3
        B^{a b}_c = A^{a b}_c
    """)
    assert set(ns_vars) == {'gDD', 'epsilonUUU', 'gdet', 'gUU', 'ADDD', 'AUUD', 'BUUD'}


def test_indexing_component() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare vD --dim 3
        w = v_{x_2}
    """)
    assert set(ns_vars) == {'vD', 'w'}
    assert str(ns_vars.w) == 'vD2'


def test_indexing_coordinate() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare coord x y z
        % declare vD --zeros --dim 3
        v_z = y^2 + 2y \\
        w = v_{x_2}
    """)
    assert set(ns_vars) == {'vD', 'y', 'w'}
    assert str(ns_vars.w) == 'y**2 + 2*y'


def test_multiple_metric() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare deltaDD --zeros --dim 3
        \delta_{ii} = 1 % noimpsum
        % declare metric gammahatDD --dim 3
        % \hat{\gamma}_{ij} = \delta_{ij}
        % declare hDD --dim 3 --sym sym01
        % declare metric gammabarDD --dim 3
        % \bar{\gamma}_{ij} = h_{ij} + \hat{\gamma}_{ij}
        % T^i_{jk} = \hat{\Gamma}^i_{jk} + \bar{\Gamma}^i_{jk}
    """)
    assert set(ns_vars) == {
        'deltaDD',
        'gammahatDD',
        'gammahatDD_dD',
        'hDD',
        'gammabarDD',
        'gammabarDD_dD',
        'gammahatdet',
        'epsilonUUU',
        'gammahatUU',
        'GammahatUDD',
        'hDD_dD',
        'gammabardet',
        'gammabarUU',
        'GammabarUDD',
        'TUDD',
    }


def test_annotation_noimpsum() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare coord r theta phi
        % declare vD --zeros --dim 3
        % v_0 = 1
        % v_1 = r
        % v_2 = r \sin{\theta}
        % R_{ij} = v_i v_j
        % declare metric gammahatDD --zeros --dim 3
        % \hat{\gamma}_{ii} = R_{ii} % noimpsum
        % declare hDD gammabarDD --dim 3 --suffix dD
        % \bar{\gamma}_{ij} = h_{ij} R_{ij} + \hat{\gamma}_{ij} % noimpsum
        T_{ijk} = \partial_k \bar{\gamma}_{ij}
    """)
    assert set(ns_vars) == {
        'gammabarDD_dD',
        'RDD',
        'r',
        'vD',
        'theta',
        'gammahatDD',
        'TDDD',
        'hDD',
        'gammabarDD',
        'hDD_dD',
    }
    assert str(ns_vars.gammahatDD) == '[[1, 0, 0], [0, r**2, 0], [0, 0, r**2*sin(theta)**2]]'
    assert (
        str(ns_vars.TDDD[0][-1])
        == '[hDD02*sin(theta) + hDD_dD020*r*sin(theta), hDD02*r*cos(theta) + hDD_dD021*r*sin(theta), hDD_dD022*r*sin(theta)]'
    )


def test_diagonal_contraction() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare hUD --dim 4
        h = h^\mu{}_\mu
    """)
    assert set(ns_vars) == {'hUD', 'h'}
    assert str(ns_vars.h) == 'hUD00 + hUD11 + hUD22 + hUD33'


def test_indexing_metric() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare metric gUU --dim 3
        % declare vD --dim 3
        % declare index mu nu --dim 3
        v^\mu = g^{\mu\nu} v_\nu
    """)
    assert set(ns_vars) == {'gUU', 'vD', 'vU'}
    assert (
        str(ns_vars.vU)
        == '[gUU00*vD0 + gUU01*vD1 + gUU02*vD2, gUU01*vD0 + gUU11*vD1 + gUU12*vD2, gUU02*vD0 + gUU12*vD1 + gUU22*vD2]'
    )


def test_inference_levi_civita() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare vU wU --dim 3
        u_i = \epsilon_{ijk} v^j w^k
    """)
    assert set(ns_vars) == {'epsilonDDD', 'vU', 'wU', 'uD'}
    assert str(ns_vars.uD) == '[vU1*wU2 - vU2*wU1, -vU0*wU2 + vU2*wU0, vU0*wU1 - vU1*wU0]'


def test_notation_covdrv() -> None:
    ns_vars1 = parse_latex_reset(r"""
        % declare FUU --dim 4 --suffix dD --sym anti01
        % declare metric gDD --dim 4 --suffix dD
        % declare k --const
        J^\mu = (4\pi k)^{-1} F^{\mu\nu}_{;\nu}
    """)
    assert set(ns_vars1) == {
        'FUU',
        'gUU',
        'gdet',
        'epsilonUUUU',
        'gDD',
        'k',
        'FUU_dD',
        'gDD_dD',
        'GammaUDD',
        'FUU_cdD',
        'JU',
    }

    ns_vars2 = parse_latex_reset(r"""
        % declare FUU --dim 4 --suffix dD --sym anti01
        % declare metric gDD --dim 4 --suffix dD
        % declare k --const
        J^\mu = (4\pi k)^{-1} \nabla_\nu F^{\mu\nu}
    """)
    assert set(ns_vars2) == {
        'FUU',
        'gUU',
        'gdet',
        'epsilonUUUU',
        'gDD',
        'k',
        'FUU_dD',
        'gDD_dD',
        'GammaUDD',
        'FUU_cdD',
        'JU',
    }

    ns_vars3 = parse_latex_reset(r"""
        % declare FUU --dim 4 --suffix dD --sym anti01
        % declare metric ghatDD --dim 4 --suffix dD
        % declare k --const
        J^\mu = (4\pi k)^{-1} \hat{\nabla}_\nu F^{\mu\nu}
    """)
    assert set(ns_vars3) == {
        'FUU',
        'ghatUU',
        'ghatdet',
        'epsilonUUUU',
        'k',
        'ghatDD',
        'FUU_dD',
        'ghatDD_dD',
        'GammahatUDD',
        'FUU_cdhatD',
        'JU',
    }


def test_schwarzschild_metric() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare coord t r theta phi
        % declare metric gDD --zeros --dim 4
        % declare G M --const
        % ignore "\begin{align}" "\end{align}"
        \begin{align}
            g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
            g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
            g_{\theta \theta} &= r^2 \\
            g_{\phi \phi} &= r^2 \sin^2{\theta}
        \end{align}
    """)
    assert str(ns_vars.gDD[0][0]) == '2*G*M/r - 1'
    assert str(ns_vars.gDD[1][1]) == '1/(-2*G*M/r + 1)'
    assert str(ns_vars.gDD[2][2]) == 'r**2'
    assert str(ns_vars.gDD[3][3]) == 'r**2*sin(theta)**2'


def test_schwarzschild_kretschmann() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare coord t r theta phi
        % declare metric gDD --zeros --dim 4
        % declare G M --const
        % ignore "\begin{align}" "\end{align}"
        \begin{align}
            g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
            g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
            g_{\theta \theta} &= r^2 \\
            g_{\phi \phi} &= r^2 \sin^2{\theta} \\
        \end{align}
        \begin{align}
            R^\alpha{}_{\beta\mu\nu} &= \partial_\mu \Gamma^\alpha_{\beta\nu} - \partial_\nu \Gamma^\alpha_{\beta\mu} + \Gamma^\alpha_{\mu\gamma}\Gamma^\gamma_{\beta\nu} - \Gamma^\alpha_{\nu\sigma}\Gamma^\sigma_{\beta\mu} \\
            K &= R^{\alpha\beta\mu\nu} R_{\alpha\beta\mu\nu} \\
            R_{\beta\nu} &= R^\alpha_{\beta\alpha\nu} \\
            R &= g^{\beta\nu} R_{\beta\nu} \\
            G_{\beta\nu} &= R_{\beta\nu} - \frac{1}{2}g_{\beta\nu}R
        \end{align}
    """)
    assert str(ns_vars.gdet) == 'r**4*(2*G*M/r - 1)*sin(theta)**2/(-2*G*M/r + 1)'
    assert ns_vars.GammaUDD[0][0][1] - ns_vars.GammaUDD[0][1][0] == 0
    assert str(ns_vars.GammaUDD[0][0][1]) == '-G*M/(r**2*(2*G*M/r - 1))'
    assert str(ns_vars.GammaUDD[1][0][0]) == 'G*M*(-2*G*M/r + 1)/r**2'
    assert str(ns_vars.GammaUDD[1][1][1]) == '-G*M/(r**2*(-2*G*M/r + 1))'
    assert str(ns_vars.GammaUDD[1][3][3]) == '-r*(-2*G*M/r + 1)*sin(theta)**2'
    assert ns_vars.GammaUDD[2][1][2] - ns_vars.GammaUDD[2][2][1] == 0
    assert str(ns_vars.GammaUDD[2][1][2]) == '1/r'
    assert str(ns_vars.GammaUDD[2][3][3]) == '-sin(theta)*cos(theta)'
    assert ns_vars.GammaUDD[2][1][3] - ns_vars.GammaUDD[2][3][1] == 0
    assert str(ns_vars.GammaUDD[3][1][3]) == '1/r'
    assert ns_vars.GammaUDD[3][2][3] - ns_vars.GammaUDD[3][3][2] == 0
    assert str(ns_vars.GammaUDD[3][2][3]) == 'cos(theta)/sin(theta)'
    assert str(sp.simplify(ns_vars.K)) == '48*G**2*M**2/r**6'
    assert sp.simplify(ns_vars.R) == 0
    for i in range(3):
        for j in range(3):
            assert sp.simplify(ns_vars.GDD[i][j]) == 0


def test_extrinsic_curvature() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare coord t r theta phi
        % declare metric gDD --zeros --dim 4
        % declare G M --const
        % ignore "\begin{align}" "\end{align}"
        \begin{align}
            g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
            g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
            g_{\theta \theta} &= r^2 \\
            g_{\phi \phi} &= r^2 \sin^2{\theta} \\
        \end{align}
        \begin{align}
            R^\alpha{}_{\beta\mu\nu} &= \partial_\mu \Gamma^\alpha_{\beta\nu} - \partial_\nu \Gamma^\alpha_{\beta\mu} + \Gamma^\alpha_{\mu\gamma}\Gamma^\gamma_{\beta\nu} - \Gamma^\alpha_{\nu\sigma}\Gamma^\sigma_{\beta\mu} \\
            K &= R^{\alpha\beta\mu\nu} R_{\alpha\beta\mu\nu} \\
            R_{\beta\nu} &= R^\alpha_{\beta\alpha\nu} \\
            R &= g^{\beta\nu} R_{\beta\nu} \\
            G_{\beta\nu} &= R_{\beta\nu} - \frac{1}{2}g_{\beta\nu}R
        \end{align}
        \begin{align}
            % declare coord r theta phi
            % declare metric gammaDD --zeros --dim 3
            \gamma_{ij} &= g_{ij} \\
            \beta_i &= g_{0 i} \\
            \alpha &= \sqrt{\gamma^{ij}\beta_i\beta_j - g_{0 0}} \\
            K_{ij} &= \frac{1}{2\alpha}\left(\nabla_i \beta_j + \nabla_j \beta_i\right) \\
            K &= \gamma^{ij} K_{ij}
        \end{align}
    """)
    for i in range(3):
        for j in range(3):
            assert ns_vars.KDD[i][j] == 0


def test_hamiltonian_momentum_contraint() -> None:
    ns_vars = parse_latex_reset(r"""
        % declare coord t r theta phi
        % declare metric gDD --zeros --dim 4
        % declare G M --const
        % ignore "\begin{align}" "\end{align}"
        \begin{align}
            g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
            g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
            g_{\theta \theta} &= r^2 \\
            g_{\phi \phi} &= r^2 \sin^2{\theta} \\
        \end{align}
        \begin{align}
            R^\alpha{}_{\beta\mu\nu} &= \partial_\mu \Gamma^\alpha_{\beta\nu} - \partial_\nu \Gamma^\alpha_{\beta\mu} + \Gamma^\alpha_{\mu\gamma}\Gamma^\gamma_{\beta\nu} - \Gamma^\alpha_{\nu\sigma}\Gamma^\sigma_{\beta\mu} \\
            K &= R^{\alpha\beta\mu\nu} R_{\alpha\beta\mu\nu} \\
            R_{\beta\nu} &= R^\alpha_{\beta\alpha\nu} \\
            R &= g^{\beta\nu} R_{\beta\nu} \\
            G_{\beta\nu} &= R_{\beta\nu} - \frac{1}{2}g_{\beta\nu}R
        \end{align}
        \begin{align}
            % declare coord r theta phi
            % declare metric gammaDD --zeros --dim 3
            \gamma_{ij} &= g_{ij} \\
            \beta_i &= g_{0 i} \\
            \alpha &= \sqrt{\gamma^{ij}\beta_i\beta_j - g_{0 0}} \\
            K_{ij} &= \frac{1}{2\alpha}\left(\nabla_i \beta_j + \nabla_j \beta_i\right) \\
            K &= \gamma^{ij} K_{ij} \\
        \end{align}
        \begin{align}
            R_{ij} &= \partial_k \Gamma^k_{ij} - \partial_j \Gamma^k_{ik}
                + \Gamma^k_{ij}\Gamma^l_{kl} - \Gamma^l_{ik}\Gamma^k_{lj} \\
            R &= \gamma^{ij} R_{ij} \\
            E &= \frac{1}{16\pi}\left(R + K^{{2}} - K_{ij}K^{ij}\right) \\
            p_i &= \frac{1}{8\pi}\left(D_j \gamma^{jk} K_{ki} - D_i K\right)
        \end{align}
    """)
    assert sp.simplify(ns_vars.E) == 0
    for i in range(3):
        assert ns_vars.pD[i] == 0


def test_inverse_covariant() -> None:
    for DIM in range(2, 5):
        ns_vars = parse_latex_reset(
            r"""
            % declare metric gDD --dim {DIM}
            % declare index latin --dim {DIM}
            T^a_c = g^{{ab}} g_{{bc}}
        """.format(DIM=DIM)
        )
        for i in range(DIM):
            for j in range(DIM):
                assert sp.simplify(ns_vars.TUD[i][j]) == (1 if i == j else 0)


def test_inverse_contravariant() -> None:
    for DIM in range(2, 5):
        ns_vars = parse_latex_reset(
            r"""
            % declare metric gUU --dim {DIM}
            % declare index latin --dim {DIM}
            T^a_c = g^{{ab}} g_{{bc}}
        """.format(DIM=DIM)
        )
        for i in range(DIM):
            for j in range(DIM):
                assert sp.simplify(ns_vars.TUD[i][j]) == (1 if i == j else 0)
