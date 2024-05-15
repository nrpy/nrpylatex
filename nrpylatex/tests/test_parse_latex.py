""" NRPyLaTeX Unit Testing """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

from nrpylatex.core.parser import Parser
import nrpylatex as nl, sympy as sp, unittest
parse_latex = lambda sentence: nl.parse_latex(sentence, reset=True)

class TestParser(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    def assert_equal(self, expr_1, expr_2):
        return self.assertEqual(sp.simplify(expr_1 - expr_2), 0)

    def test_expression_arithmetic(self):
        expr = r'-(\frac{2}{3} + 2\sqrt[5]{x + 3})'
        self.assertEqual(
            str(parse_latex(expr)),
            '-2*(x + 3)**(1/5) - 2/3'
        )

    def test_expression_exponential(self):
        expr = r'e^{\ln{x}} - \tanh{xy}'
        self.assertEqual(
            str(parse_latex(expr)),
            'x - tanh(x*y)'
        )

    def test_expression_trigonometric(self):
        expr = r'x\cos{\pi} + \sin{\sin^{-1}{y}}'
        self.assertEqual(
            str(parse_latex(expr)),
            '-x + y'
        )

    def test_expression_derivative(self):
        expr = r'\partial_x (x^2 + 2x)'
        self.assertEqual(
            str(parse_latex(expr).doit()),
            '2*x + 2'
        )

    def test_generation_covdrv(self):
        function = sp.Function('Tensor')(sp.Symbol('T'))
        self.assertEqual(
            nl.Generator.generate_covdrv(function, 'beta'),
            r'\nabla_{\beta} T = \partial_{\beta} T'
        )
        function = sp.Function('Tensor')(sp.Symbol('TUU'), sp.Symbol('mu'), sp.Symbol('nu'))
        self.assertEqual(
            nl.Generator.generate_covdrv(function, 'beta'),
            r'\nabla_{\beta} T^{\mu \nu} = \partial_{\beta} T^{\mu \nu} + \mathrm{Gamma}^{\mu}_{i_1 \beta} (T^{i_1 \nu}) + \mathrm{Gamma}^{\nu}_{i_1 \beta} (T^{\mu i_1})'
        )
        function = sp.Function('Tensor')(sp.Symbol('TUD'), sp.Symbol('mu'), sp.Symbol('nu'))
        self.assertEqual(
            nl.Generator.generate_covdrv(function, 'beta'),
            r'\nabla_{\beta} T^{\mu}_{\nu} = \partial_{\beta} T^{\mu}_{\nu} + \mathrm{Gamma}^{\mu}_{i_1 \beta} (T^{i_1}_{\nu}) - \mathrm{Gamma}^{i_1}_{\nu \beta} (T^{\mu}_{i_1})'
        )
        function = sp.Function('Tensor')(sp.Symbol('TDD'), sp.Symbol('mu'), sp.Symbol('nu'))
        self.assertEqual(
            nl.Generator.generate_covdrv(function, 'beta'),
            r'\nabla_{\beta} T_{\mu \nu} = \partial_{\beta} T_{\mu \nu} - \mathrm{Gamma}^{i_1}_{\mu \beta} (T_{i_1 \nu}) - \mathrm{Gamma}^{i_1}_{\nu \beta} (T_{\mu i_1})'
        )

    def test_generation_nested_covdrv(self):
        parse_latex(r"""
            % declare metric gDD --dim 4 --suffix dD
            % declare vU --dim 4 --suffix dD
            T^\mu_\nu = \nabla_\nu v^\mu
        """)
        function = sp.Function('Tensor')(sp.Symbol('vU_cdD'), sp.Symbol('mu'), sp.Symbol('nu'))
        self.assertEqual(
            nl.Generator.generate_covdrv(function, 'beta'),
            r'\nabla_{\beta} \nabla_{\nu} v^{\mu} = \partial_{\beta} \nabla_{\nu} v^{\mu} + \mathrm{Gamma}^{\mu}_{i_1 \beta} (\nabla_{\nu} v^{i_1}) - \mathrm{Gamma}^{i_1}_{\nu \beta} (\nabla_{i_1} v^{\mu})'
        )

    def test_generation_liedrv(self):
        function = sp.Function('Tensor')(sp.Symbol('g'))
        self.assertEqual(
            nl.Generator.generate_liedrv(function, 'beta', 2),
            r'\mathcal{L}_\mathrm{beta} g = \mathrm{beta}^{i_1} \partial_{i_1} g + (2)(\partial_{i_1} \mathrm{beta}^{i_1}) g'
        )
        function = sp.Function('Tensor')(sp.Symbol('gUU'), sp.Symbol('i'), sp.Symbol('j'))
        self.assertEqual(
            nl.Generator.generate_liedrv(function, 'beta'),
            r'\mathcal{L}_\mathrm{beta} g^{i j} = \mathrm{beta}^{i_1} \partial_{i_1} g^{i j} - (\partial_{i_1} \mathrm{beta}^{i}) g^{i_1 j} - (\partial_{i_1} \mathrm{beta}^{j}) g^{i i_1}'
        )
        function = sp.Function('Tensor')(sp.Symbol('gUD'), sp.Symbol('i'), sp.Symbol('j'))
        self.assertEqual(
            nl.Generator.generate_liedrv(function, 'beta'),
            r'\mathcal{L}_\mathrm{beta} g^{i}_{j} = \mathrm{beta}^{i_1} \partial_{i_1} g^{i}_{j} - (\partial_{i_1} \mathrm{beta}^{i}) g^{i_1}_{j} + (\partial_{j} \mathrm{beta}^{i_1}) g^{i}_{i_1}'
        )
        function = sp.Function('Tensor')(sp.Symbol('gDD'), sp.Symbol('i'), sp.Symbol('j'))
        self.assertEqual(
            nl.Generator.generate_liedrv(function, 'beta'),
            r'\mathcal{L}_\mathrm{beta} g_{i j} = \mathrm{beta}^{i_1} \partial_{i_1} g_{i j} + (\partial_{i} \mathrm{beta}^{i_1}) g_{i_1 j} + (\partial_{j} \mathrm{beta}^{i_1}) g_{i i_1}'
        )

    def test_replacement_rule(self):
        nl.parse_latex(r"""
            % replace "\1'" -> "\mathrm{\1prime}"
            % replace "\1_{\2*}" -> "\mathrm{\1_\2*}"
            % replace "\1_\2"    -> "\mathrm{\1_\2}"
            % replace "\1^{\2*}" -> "\1^{{\2*}}"
            % replace "\1^\2"    -> "\1^{{\2}}"
        """)
        expr = r"x_n^4 + x'_n \exp{x_n y_n^2}"
        self.assertEqual(
            str(nl.parse_latex(expr)),
            "x_n**4 + xprime_n*exp(x_n*y_n**2)"
        )
        Parser.initialize(reset=True)

    def test_recursive_replacement(self):
        nl.parse_latex(r"""
            % replace "K" -> "\mathrm{trK}"
            x = K^{{2}} \\
            y = \mathrm{trK} + x
        """)
        expr = r"K^{{2}} + \mathrm{trK}"
        self.assertEqual(
            str(nl.parse_latex(expr)),
            "trK**2 + trK"
        )
        Parser.initialize(reset=True)

    def test_product_rule(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare index latin --dim 2
                % declare vU wU --dim 2 --suffix dD
                T^{ab}_c = \partial_c (v^a w^b)
            """)),
            {'vU', 'wU', 'vU_dD', 'wU_dD', 'TUUD'}
        )
        self.assertEqual(str(TUUD[0][0][0]),
            'vU0*wU_dD00 + vU_dD00*wU0'
        )

    def test_upwind_suffix(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare index latin --dim 2
                % declare vU --dim 2 --suffix dD
                % declare w --const
                T^a_c = % suffix dupD
                \partial_c (v^a w)
            """)),
            {'w', 'vU', 'vU_dupD', 'TUD'}
        )
        self.assertEqual(str(TUD),
            '[[vU_dupD00*w, vU_dupD01*w], [vU_dupD10*w, vU_dupD11*w]]'
        )

    def test_inference_covdrv(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare index latin --dim 4
                % declare metric gDD --dim 4 --suffix dD
                % declare vU --dim 4 --suffix dD
                T^{ab} = \nabla^b v^a
            """)),
            {'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'vU', 'vU_dD', 'gDD_dD', 'GammaUDD', 'vU_cdD', 'vU_cdU', 'TUU'}
        )

    def test_inference_pardrv(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare coord x y
                % declare index latin --dim 2
                % declare uD --zeros --dim 2
                u_x = x^2 + 2x \\
                u_y = y\sqrt{x}
                % declare wD vD --dim 2 --suffix dD
                v_a = u_a + w_a \\
                T_{ab} = \partial_b v_a
            """)),
            {'x', 'y', 'uD', 'wD', 'vD', 'vD_dD', 'wD_dD', 'TDD'}
        )
        self.assertEqual(str(TDD),
            '[[wD_dD00 + 2*x + 2, wD_dD01], [wD_dD10 + y/(2*sqrt(x)), wD_dD11 + sqrt(x)]]'
        )

    def test_notation_pardrv(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare index latin --dim 2
                % declare vD uD wD --dim 2 --suffix dD
                T_{abc} = ((v_a + u_a)_{,b} - w_{a,b})_{,c}
            """)),
            {'vD', 'uD', 'wD', 'TDDD', 'uD_dD', 'vD_dD', 'wD_dD', 'wD_dDD', 'uD_dDD', 'vD_dDD'}
        )
        self.assertEqual(str(TDDD[0][0][0]),
            'uD_dDD000 + vD_dDD000 - wD_dDD000'
        )

    def test_spherical_riemann(self):
        parse_latex(r"""
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
        self.assertEqual(str(GammaUDD[0][1][1]),
            '-sin(theta)*cos(theta)'
        )
        self.assertEqual(GammaUDD[1][0][1] - GammaUDD[1][1][0], 0)
        self.assertEqual(str(GammaUDD[1][0][1]),
            'cos(theta)/sin(theta)'
        )
        self.assertEqual(RDDDD[0][1][0][1] - (-RDDDD[0][1][1][0]) + (-RDDDD[1][0][0][1]) - RDDDD[1][0][1][0], 0)
        self.assertEqual(str(RDDDD[0][1][0][1]),
            'r**2*sin(theta)**2'
        )
        self.assertEqual(RDD[0][0], 1)
        self.assertEqual(str(RDD[1][1]),
            'sin(theta)**2'
        )
        self.assertEqual(RDD[0][1] - RDD[1][0], 0)
        self.assertEqual(RDD[0][1], 0)
        self.assertEqual(str(sp.simplify(R)),
            '2/r**2'
        )

    def test_dimension_reduction(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare gDD --dim 4 --sym sym01
                \gamma_{ij} = g_{ij}
            """)),
            {'gDD', 'gammaDD'}
        )
        self.assertEqual(str(gammaDD),
            '[[gDD11, gDD12, gDD13], [gDD12, gDD22, gDD23], [gDD13, gDD23, gDD33]]'
        )

    def test_spatial_contraction(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare TUU --dim 3
                % declare vD --dim 2
                % declare index i --dim 2
                w^a = T^{a i} v_i
            """)),
            {'TUU', 'vD', 'wU'}
        )
        self.assertEqual(str(wU),
            '[TUU01*vD0 + TUU02*vD1, TUU11*vD0 + TUU12*vD1, TUU21*vD0 + TUU22*vD1]'
        )

    def test_inference_indexing(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare metric gDD --dim 3
                % declare ADDD --dim 3
                B^{a b}_c = A^{a b}_c
            """)),
            {'gDD', 'epsilonUUU', 'gdet', 'gUU', 'ADDD', 'AUUD', 'BUUD'}
        )

    def test_indexing_component(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare vD --dim 3
                w = v_{x_2}
            """)),
            {'vD', 'w'}
        )
        self.assertEqual(str(w),
            'vD2'
        )

    def test_indexing_coordinate(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare coord x y z
                % declare vD --zeros --dim 3
                v_z = y^2 + 2y \\
                w = v_{x_2}
            """)),
            {'vD', 'y', 'w'}
        )
        self.assertEqual(str(w),
            'y**2 + 2*y'
        )

    def test_multiple_metric(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare deltaDD --zeros --dim 3
                \delta_{ii} = 1 % noimpsum
                % declare metric gammahatDD --dim 3
                % \hat{\gamma}_{ij} = \delta_{ij}
                % declare hDD --dim 3 --sym sym01
                % declare metric gammabarDD --dim 3
                % \bar{\gamma}_{ij} = h_{ij} + \hat{\gamma}_{ij}
                % T^i_{jk} = \hat{\Gamma}^i_{jk} + \bar{\Gamma}^i_{jk}
            """)),
            {'deltaDD', 'gammahatDD', 'gammahatDD_dD', 'hDD', 'gammabarDD', 'gammabarDD_dD', 'gammahatdet', 'epsilonUUU', 'gammahatUU', 'GammahatUDD', 'hDD_dD', 'gammabardet', 'gammabarUU', 'GammabarUDD', 'TUDD'}
        )

    def test_annotation_noimpsum(self):
        self.assertEqual(
            set(parse_latex(r"""
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
            """)),
            {'gammabarDD_dD', 'RDD', 'r', 'vD', 'theta', 'gammahatDD', 'TDDD', 'hDD', 'gammabarDD', 'hDD_dD'}
        )
        self.assertEqual(str(gammahatDD),
            '[[1, 0, 0], [0, r**2, 0], [0, 0, r**2*sin(theta)**2]]'
        )
        self.assertEqual(str(TDDD[0][-1]),
            '[hDD02*sin(theta) + hDD_dD020*r*sin(theta), hDD02*r*cos(theta) + hDD_dD021*r*sin(theta), hDD_dD022*r*sin(theta)]'
        )

    def test_diagonal_contraction(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare hUD --dim 4
                h = h^\mu{}_\mu
            """)),
            {'hUD', 'h'}
        )
        self.assertEqual(str(h),
            'hUD00 + hUD11 + hUD22 + hUD33'
        )

    def test_indexing_metric(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare metric gUU --dim 3
                % declare vD --dim 3
                % declare index mu nu --dim 3
                v^\mu = g^{\mu\nu} v_\nu
            """)),
            {'gUU', 'vD', 'vU'}
        )
        self.assertEqual(str(vU),
            '[gUU00*vD0 + gUU01*vD1 + gUU02*vD2, gUU01*vD0 + gUU11*vD1 + gUU12*vD2, gUU02*vD0 + gUU12*vD1 + gUU22*vD2]'
        )

    def test_inference_levi_civita(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare vU wU --dim 3
                u_i = \epsilon_{ijk} v^j w^k
            """)),
            {'epsilonDDD', 'vU', 'wU', 'uD'}
        )
        self.assertEqual(str(uD),
            '[vU1*wU2 - vU2*wU1, -vU0*wU2 + vU2*wU0, vU0*wU1 - vU1*wU0]'
        )

    def test_notation_covdrv(self):
        self.assertEqual(
            set(parse_latex(r"""
                % declare FUU --dim 4 --suffix dD --sym anti01
                % declare metric gDD --dim 4 --suffix dD
                % declare k --const
                J^\mu = (4\pi k)^{-1} F^{\mu\nu}_{;\nu}
            """)),
            {'FUU', 'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'k', 'FUU_dD', 'gDD_dD', 'GammaUDD', 'FUU_cdD', 'JU'}
        )
        self.assertEqual(
            set(parse_latex(r"""
                % declare FUU --dim 4 --suffix dD --sym anti01
                % declare metric gDD --dim 4 --suffix dD
                % declare k --const
                J^\mu = (4\pi k)^{-1} \nabla_\nu F^{\mu\nu}
            """)),
            {'FUU', 'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'k', 'FUU_dD', 'gDD_dD', 'GammaUDD', 'FUU_cdD', 'JU'}
        )
        self.assertEqual(
            set(parse_latex(r"""
                % declare FUU --dim 4 --suffix dD --sym anti01
                % declare metric ghatDD --dim 4 --suffix dD
                % declare k --const
                J^\mu = (4\pi k)^{-1} \hat{\nabla}_\nu F^{\mu\nu}
            """)),
            {'FUU', 'ghatUU', 'ghatdet', 'epsilonUUUU', 'k',  'ghatDD', 'FUU_dD', 'ghatDD_dD', 'GammahatUDD', 'FUU_cdhatD', 'JU'}
        )

    def test_schwarzschild_metric(self):
        parse_latex(r"""
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
        self.assertEqual(str(gDD[0][0]),
            '2*G*M/r - 1'
        )
        self.assertEqual(str(gDD[1][1]),
            '1/(-2*G*M/r + 1)'
        )
        self.assertEqual(str(gDD[2][2]),
            'r**2'
        )
        self.assertEqual(str(gDD[3][3]),
            'r**2*sin(theta)**2'
        )

    def test_schwarzschild_kretschmann(self):
        parse_latex(r"""
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
        self.assertEqual(str(gdet),
            'r**4*(2*G*M/r - 1)*sin(theta)**2/(-2*G*M/r + 1)'
        )
        self.assertEqual(GammaUDD[0][0][1] - GammaUDD[0][1][0], 0)
        self.assertEqual(str(GammaUDD[0][0][1]),
            '-G*M/(r**2*(2*G*M/r - 1))'
        )
        self.assertEqual(str(GammaUDD[1][0][0]),
            'G*M*(-2*G*M/r + 1)/r**2'
        )
        self.assertEqual(str(GammaUDD[1][1][1]),
            '-G*M/(r**2*(-2*G*M/r + 1))'
        )
        self.assertEqual(str(GammaUDD[1][3][3]),
            '-r*(-2*G*M/r + 1)*sin(theta)**2'
        )
        self.assertEqual(GammaUDD[2][1][2] - GammaUDD[2][2][1], 0)
        self.assertEqual(str(GammaUDD[2][1][2]),
            '1/r'
        )
        self.assertEqual(str(GammaUDD[2][3][3]),
            '-sin(theta)*cos(theta)'
        )
        self.assertEqual(GammaUDD[2][1][3] - GammaUDD[2][3][1], 0)
        self.assertEqual(str(GammaUDD[3][1][3]),
            '1/r'
        )
        self.assertEqual(GammaUDD[3][2][3] - GammaUDD[3][3][2], 0)
        self.assertEqual(str(GammaUDD[3][2][3]),
            'cos(theta)/sin(theta)'
        )
        self.assertEqual(str(sp.simplify(K)),
            '48*G**2*M**2/r**6'
        )
        self.assertEqual(sp.simplify(R), 0)
        for i in range(3):
            for j in range(3):
                self.assertEqual(sp.simplify(GDD[i][j]), 0)

    def test_extrinsic_curvature(self):
        parse_latex(r"""
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
                self.assertEqual(KDD[i][j], 0)

    def test_hamiltonian_momentum_contraint(self):
        parse_latex(r"""
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
        self.assertEqual(sp.simplify(E), 0)
        for i in range(3):
            self.assertEqual(pD[i], 0)

    def test_inverse_covariant(self):
        for DIM in range(2, 5):
            parse_latex(r"""
                % declare metric gDD --dim {DIM}
                % declare index latin --dim {DIM}
                T^a_c = g^{{ab}} g_{{bc}}
            """.format(DIM=DIM))
            for i in range(DIM):
                for j in range(DIM):
                    self.assertEqual(sp.simplify(TUD[i][j]), 1 if i == j else 0)
    
    def test_inverse_contravariant(self):
        for DIM in range(2, 5):
            parse_latex(r"""
                % declare metric gUU --dim {DIM}
                % declare index latin --dim {DIM}
                T^a_c = g^{{ab}} g_{{bc}}
            """.format(DIM=DIM))
            for i in range(DIM):
                for j in range(DIM):
                    self.assertEqual(sp.simplify(TUD[i][j]), 1 if i == j else 0)

if __name__ == '__main__':
    unittest.main()
