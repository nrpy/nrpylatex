""" parse_latex.py Unit Testing """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

# pylint: disable = import-error, protected-access, exec-used
from nrpylatex.core.assert_equal import assert_equal
import nrpylatex as nl, sympy as sp, unittest
parse_latex = lambda sentence: nl.parse_latex(sentence, reset=True, ignore_warning=True)

class TestParser(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    def test_expression_1(self):
        expr = r'-(\frac{2}{3} + 2\sqrt[5]{x + 3})'
        self.assertEqual(
            str(parse_latex(expr)),
            '-2*(x + 3)**(1/5) - 2/3'
        )

    def test_expression_2(self):
        expr = r'e^{\ln x} + \sin(\sin^{-1} y) - \tanh(xy)'
        self.assertEqual(
            str(parse_latex(expr)),
            'x + y - tanh(x*y)'
        )

    def test_expression_3(self):
        expr = r'\partial_x (x^2 + 2x)'
        self.assertEqual(
            str(parse_latex(expr).doit()),
            '2*x + 2'
        )

    def test_expression_4(self):
        function = sp.Function('Tensor')(sp.Symbol('T'))
        self.assertEqual(
            nl.Parser._generate_covdrv(function, 'beta'),
            r'\nabla_{\beta} T = \partial_{\beta} T'
        )
        function = sp.Function('Tensor')(sp.Symbol('TUU'), sp.Symbol('mu'), sp.Symbol('nu'))
        self.assertEqual(
            nl.Parser._generate_covdrv(function, 'beta'),
            r'\nabla_{\beta} T^{\mu \nu} = \partial_{\beta} T^{\mu \nu} + \text{Gamma}^{\mu}_{a \beta} (T^{a \nu}) + \text{Gamma}^{\nu}_{a \beta} (T^{\mu a})'
        )
        function = sp.Function('Tensor')(sp.Symbol('TUD'), sp.Symbol('mu'), sp.Symbol('nu'))
        self.assertEqual(
            nl.Parser._generate_covdrv(function, 'beta'),
            r'\nabla_{\beta} T^{\mu}_{\nu} = \partial_{\beta} T^{\mu}_{\nu} + \text{Gamma}^{\mu}_{a \beta} (T^{a}_{\nu}) - \text{Gamma}^{a}_{\nu \beta} (T^{\mu}_{a})'
        )
        function = sp.Function('Tensor')(sp.Symbol('TDD'), sp.Symbol('mu'), sp.Symbol('nu'))
        self.assertEqual(
            nl.Parser._generate_covdrv(function, 'beta'),
            r'\nabla_{\beta} T_{\mu \nu} = \partial_{\beta} T_{\mu \nu} - \text{Gamma}^{a}_{\mu \beta} (T_{a \nu}) - \text{Gamma}^{a}_{\nu \beta} (T_{\mu a})'
        )

    def test_expression_5(self):
        parse_latex(r"""
            % vardef -diff_suffix=dD -metric gDD::4D
            % vardef -diff_suffix=dD vU::4D
            T^\mu_b = \nabla_b v^\mu
        """)
        function = sp.Function('Tensor')(sp.Symbol('vU_cdD'), sp.Symbol('mu'), sp.Symbol('b'))
        self.assertEqual(
            nl.Parser._generate_covdrv(function, 'a'),
            r'\nabla_{a} \nabla_{b} v^{\mu} = \partial_{a} \nabla_{b} v^{\mu} + \text{Gamma}^{\mu}_{c a} (\nabla_{b} v^{c}) - \text{Gamma}^{c}_{b a} (\nabla_{c} v^{\mu})'
        )

    def test_expression_6(self):
        function = sp.Function('Tensor')(sp.Symbol('g'))
        self.assertEqual(
            nl.Parser._generate_liedrv(function, 'beta', 2),
            r'\mathcal{L}_\text{beta} g = \text{beta}^{a} \partial_{a} g + (2)(\partial_{a} \text{beta}^{a}) g'
        )
        function = sp.Function('Tensor')(sp.Symbol('gUU'), sp.Symbol('i'), sp.Symbol('j'))
        self.assertEqual(
            nl.Parser._generate_liedrv(function, 'beta'),
            r'\mathcal{L}_\text{beta} g^{i j} = \text{beta}^{a} \partial_{a} g^{i j} - (\partial_{a} \text{beta}^{i}) g^{a j} - (\partial_{a} \text{beta}^{j}) g^{i a}'
        )
        function = sp.Function('Tensor')(sp.Symbol('gUD'), sp.Symbol('i'), sp.Symbol('j'))
        self.assertEqual(
            nl.Parser._generate_liedrv(function, 'beta'),
            r'\mathcal{L}_\text{beta} g^{i}_{j} = \text{beta}^{a} \partial_{a} g^{i}_{j} - (\partial_{a} \text{beta}^{i}) g^{a}_{j} + (\partial_{j} \text{beta}^{a}) g^{i}_{a}'
        )
        function = sp.Function('Tensor')(sp.Symbol('gDD'), sp.Symbol('i'), sp.Symbol('j'))
        self.assertEqual(
            nl.Parser._generate_liedrv(function, 'beta'),
            r'\mathcal{L}_\text{beta} g_{i j} = \text{beta}^{a} \partial_{a} g_{i j} + (\partial_{i} \text{beta}^{a}) g_{a j} + (\partial_{j} \text{beta}^{a}) g_{i a}'
        )

    def test_srepl_macro(self):
        nl.parse_latex(r"""
            % srepl -persist "<1>'" -> "\text{<1>prime}"
            % srepl -persist "\text{<1..>}_<2>" -> "\text{(<1..>)<2>}"
            % srepl -persist "<1>_{<2>}" -> "<1>_<2>", "<1>_<2>" -> "\text{<1>_<2>}"
            % srepl -persist "\text{(<1..>)<2>}" -> "\text{<1..>_<2>}"
            % srepl -persist "<1>^{<2>}" -> "<1>^<2>", "<1>^<2>" -> "<1>^{{<2>}}"
        """)
        expr = r"x_n^4 + x'_n \exp(x_n y_n^2)"
        self.assertEqual(
            str(nl.parse_latex(expr)),
            "x_n**4 + xprime_n*exp(x_n*y_n**2)"
        )
        parse_latex(r""" % srepl -persist "<1>'^{<2..>}" -> "\text{<1>prime}" """)
        expr = r"v'^{label}"
        self.assertEqual(
            str(nl.parse_latex(expr)),
            "vprime"
        )

    def test_assignment_1(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -diff_suffix=dD vU::2D, wU::2D
                % attrib index [a-z]::2D
                T^{ab}_c = \partial_c (v^a w^b)
            """)),
            {'vU', 'wU', 'vU_dD', 'wU_dD', 'TUUD'}
        )
        self.assertEqual(str(TUUD[0][0][0]),
            'vU0*wU_dD00 + vU_dD00*wU0'
        )

    def test_assignment_2(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -diff_suffix=dD vU::2D
                % assign -const w
                % attrib index [a-z]::2D
                T^a_c = % diff_suffix dupD
                \partial_c (v^a w)
            """)),
            {'w', 'vU', 'vU_dupD', 'TUD'}
        )
        self.assertEqual(str(TUD),
            '[[vU_dupD00*w, vU_dupD01*w], [vU_dupD10*w, vU_dupD11*w]]'
        )

    def test_assignment_3(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -diff_suffix=dD -metric gDD::4D
                % vardef -diff_suffix=dD vU::4D
                % attrib index [a-z]::4D
                T^{ab} = \nabla^b v^a
            """)),
            {'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'vU', 'vU_dD', 'gDD_dD', 'GammaUDD', 'vU_cdD', 'vU_cdU', 'TUU'}
        )

    def test_assignment_4(self):
        self.assertEqual(
            set(parse_latex(r"""
                % attrib coord [x, y]
                % vardef uD::2D, wD::2D
                % attrib index [a-z]::2D
                u_x = x^2 + 2x \\
                u_y = y\sqrt{x} \\
                v_a = u_a + w_a \\
                % assign -diff_suffix=dD wD, vD
                T_{ab} = \partial_b v_a
            """)),
            {'x', 'y', 'uD', 'wD', 'vD', 'vD_dD', 'wD_dD', 'TDD'}
        )
        self.assertEqual(str(TDD),
            '[[wD_dD00 + 2*x + 2, wD_dD01], [wD_dD10 + y/(2*sqrt(x)), wD_dD11 + sqrt(x)]]'
        )

    def test_assignment_5(self):
        self.assertEqual(
            set(parse_latex(r"""
                    % vardef -diff_suffix=dD vD::2D, uD::2D, wD::2D
                    % attrib index [a-z]::2D
                    T_{abc} = ((v_a + u_a)_{,b} - w_{a,b})_{,c}
            """)),
            {'vD', 'uD', 'wD', 'TDDD', 'uD_dD', 'vD_dD', 'wD_dD', 'wD_dDD', 'uD_dDD', 'vD_dDD'}
        )
        self.assertEqual(str(TDDD[0][0][0]),
            'uD_dDD000 + vD_dDD000 - wD_dDD000'
        )

    def test_assignment_6(self):
        parse_latex(r"""
            % attrib coord [\theta, \phi]
            % vardef -zero gDD::2D
            % assign -const r
            % attrib index [a-z]::2D
            \begin{align*}
                g_{0 0} &= r^2 \\
                g_{1 1} &= r^2 \sin^2(\theta)
            \end{align*}
            % assign -metric gDD
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
        assert_equal(GammaUDD[1][0][1] - GammaUDD[1][1][0], 0, suppress_message=True)
        self.assertEqual(str(GammaUDD[1][0][1]),
            'cos(theta)/sin(theta)'
        )
        assert_equal(RDDDD[0][1][0][1] - (-RDDDD[0][1][1][0]) + (-RDDDD[1][0][0][1]) - RDDDD[1][0][1][0], 0, suppress_message=True)
        self.assertEqual(str(RDDDD[0][1][0][1]),
            'r**2*sin(theta)**2'
        )
        assert_equal(RDD[0][0], 1, suppress_message=True)
        self.assertEqual(str(RDD[1][1]),
            'sin(theta)**2'
        )
        assert_equal(RDD[0][1] - RDD[1][0], 0, suppress_message=True)
        assert_equal(RDD[0][1], 0, suppress_message=True)
        self.assertEqual(str(R),
            '2/r**2'
        )

    def test_assignment_7(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -symmetry=sym01 gDD::4D
                \gamma_{ij} = g_{ij}
            """)),
            {'gDD', 'gammaDD'}
        )
        self.assertEqual(str(gammaDD),
            '[[gDD11, gDD12, gDD13], [gDD12, gDD22, gDD23], [gDD13, gDD23, gDD33]]'
        )

    def test_assignment_8(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef TUU::3D
                % vardef vD::2D
                % attrib index i::2D
                w^a = T^{a i} v_i
            """)),
            {'TUU', 'vD', 'wU'}
        )
        self.assertEqual(str(wU),
            '[TUU01*vD0 + TUU02*vD1, TUU11*vD0 + TUU12*vD1, TUU21*vD0 + TUU22*vD1]'
        )

    def test_assignment_9(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -metric gDD::3D
                % vardef ADDD::3D, AUUU::3D
                B^{a b}_c = A^{a b}_c
            """)),
            {'gDD', 'epsilonUUU', 'gdet', 'gUU', 'GammaUDD', 'ADDD', 'AUUU', 'AUUD', 'BUUD'}
        )

    def test_assignment_10(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef vD::3D
                w = v_{x_2}
            """)),
            {'vD', 'w'}
        )
        self.assertEqual(str(w),
            'vD2'
        )

    def test_assignment_11(self):
        self.assertEqual(
            set(parse_latex(r"""
                % attrib coord [x, y, z]
                % vardef -zero vD::3D
                v_z = y^2 + 2y \\
                w = v_{x_2}
            """)),
            {'vD', 'y', 'w'}
        )
        self.assertEqual(str(w),
            'y**2 + 2*y'
        )

    def test_assignment_12(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -kron deltaDD::3D
                % parse \hat{\gamma}_{ij} = \delta_{ij}
                % assign -diff_suffix=none -metric gammahatDD
                % vardef -diff_suffix=dD -symmetry=sym01 hDD::3D
                % parse \bar{\gamma}_{ij} = h_{ij} + \hat{\gamma}_{ij}
                % assign -diff_suffix=dD -metric gammabarDD
            """)),
            {'gammabardet', 'GammabarUDD', 'gammabarDD', 'gammabarDD_dD', 'gammahatdet', 'hDD', 'GammahatUDD', 'hDD_dD', 'gammahatUU', 'deltaDD', 'gammabarUU', 'epsilonUUU', 'gammahatDD'}
        )

    def test_assignment_13(self):
        self.assertEqual(
            set(parse_latex(r"""
                % attrib coord [r, \theta, \phi]
                % vardef vD::3D
                % parse v_0 = 1
                % parse v_1 = r
                % parse v_2 = r \sin \theta
                % parse R_{ij} = v_i v_j
                % vardef -zero gammahatDD::3D
                % parse \hat{\gamma}_{ii} = R_{ii} % noimpsum
                % vardef -diff_suffix=dD hDD::3D
                % parse \bar{\gamma}_{ij} = h_{ij} R_{ij} + \hat{\gamma}_{ij} % noimpsum
                % assign -diff_suffix=dD gammabarDD
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

    def test_example_1(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef hUD::4D
                h = h^\mu{}_\mu
            """)),
            {'hUD', 'h'}
        )
        self.assertEqual(str(h),
            'hUD00 + hUD11 + hUD22 + hUD33'
        )

    def test_example_2(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -metric gUU::3D
                % vardef vD::3D
                v^\mu = g^{\mu\nu} v_\nu
            """)),
            {'gUU', 'epsilonDDD', 'gdet', 'gDD', 'GammaUDD', 'vD', 'vU'}
        )
        self.assertEqual(str(vU),
            '[gUU00*vD0 + gUU01*vD1 + gUU02*vD2, gUU01*vD0 + gUU11*vD1 + gUU12*vD2, gUU02*vD0 + gUU12*vD1 + gUU22*vD2]'
        )

    def test_example_3(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef vU::3D, wU::3D
                u_i = \epsilon_{ijk} v^j w^k
            """)),
            {'epsilonDDD', 'vU', 'wU', 'uD'}
        )
        self.assertEqual(str(uD),
            '[vU1*wU2 - vU2*wU1, -vU0*wU2 + vU2*wU0, vU0*wU1 - vU1*wU0]'
        )

    def test_example_4(self):
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -diff_suffix=dD -symmetry=anti01 FUU::4D
                % vardef -diff_suffix=dD -metric gDD::4D
                % assign -const k
                J^\mu = (4\pi k)^{-1} F^{\mu\nu}_{;\nu}
            """)),
            {'FUU', 'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'k', 'FUU_dD', 'gDD_dD', 'GammaUDD', 'FUU_cdD', 'JU'}
        )
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -diff_suffix=dD -symmetry=anti01 FUU::4D
                % vardef -diff_suffix=dD -metric gDD::4D
                % assign -const k
                J^\mu = (4\pi k)^{-1} \nabla_\nu F^{\mu\nu}
            """)),
            {'FUU', 'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'k', 'FUU_dD', 'gDD_dD', 'GammaUDD', 'FUU_cdD', 'JU'}
        )
        self.assertEqual(
            set(parse_latex(r"""
                % vardef -diff_suffix=dD -symmetry=anti01 FUU::4D
                % vardef -diff_suffix=dD -metric ghatDD::4D
                % assign -const k
                J^\mu = (4\pi k)^{-1} \hat{\nabla}_\nu F^{\mu\nu}
            """)),
            {'FUU', 'ghatUU', 'ghatdet', 'epsilonUUUU', 'k',  'ghatDD', 'FUU_dD', 'ghatDD_dD', 'GammahatUDD', 'FUU_cdhatD', 'JU'}
        )

    def test_example_5_1(self):
        nl.parse_latex(r"""
            % attrib coord [t, r, \theta, \phi]
            % vardef -zero gDD::4D
            % assign -const G, M
            \begin{align}
                g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
                g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
                g_{\theta \theta} &= r^2 \\
                g_{\phi \phi} &= r^2 \sin^2\theta
            \end{align}
            % assign -metric gDD
        """, ignore_warning=True)
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
        self.assertEqual(str(gdet),
            'r**4*(2*G*M/r - 1)*sin(theta)**2/(-2*G*M/r + 1)'
        )

    def test_example_5_2(self):
        nl.parse_latex(r"""
            \begin{align}
                R^\alpha{}_{\beta\mu\nu} &= \partial_\mu \Gamma^\alpha_{\beta\nu} - \partial_\nu \Gamma^\alpha_{\beta\mu} + \Gamma^\alpha_{\mu\gamma}\Gamma^\gamma_{\beta\nu} - \Gamma^\alpha_{\nu\sigma}\Gamma^\sigma_{\beta\mu} \\
                K &= R^{\alpha\beta\mu\nu} R_{\alpha\beta\mu\nu} \\
                R_{\beta\nu} &= R^\alpha_{\beta\alpha\nu} \\
                R &= g^{\beta\nu} R_{\beta\nu} \\
                G_{\beta\nu} &= R_{\beta\nu} - \frac{1}{2}g_{\beta\nu}R
            \end{align}
        """, ignore_warning=True)
        assert_equal(GammaUDD[0][0][1] - GammaUDD[0][1][0], 0, suppress_message=True)
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
        assert_equal(GammaUDD[2][1][2] - GammaUDD[2][2][1], 0, suppress_message=True)
        self.assertEqual(str(GammaUDD[2][1][2]),
            '1/r'
        )
        self.assertEqual(str(GammaUDD[2][3][3]),
            '-sin(theta)*cos(theta)'
        )
        assert_equal(GammaUDD[2][1][3] - GammaUDD[2][3][1], 0, suppress_message=True)
        self.assertEqual(str(GammaUDD[3][1][3]),
            '1/r'
        )
        assert_equal(GammaUDD[3][2][3] - GammaUDD[3][3][2], 0, suppress_message=True)
        self.assertEqual(str(GammaUDD[3][2][3]),
            'cos(theta)/sin(theta)'
        )
        self.assertEqual(str(sp.simplify(K)),
            '48*G**2*M**2/r**6'
        )
        assert_equal(R, 0, suppress_message=True)
        for i in range(3):
            for j in range(3):
                assert_equal(GDD[i][j], 0, suppress_message=True)

    @staticmethod
    def test_example_6_1():
        nl.parse_latex(r"""
            % attrib coord [r, \theta, \phi]
            \begin{align}
                \gamma_{ij} &= g_{ij} \\
                % assign -metric gammaDD
                \beta_i &= g_{0 i} \\
                \alpha &= \sqrt{\gamma^{ij}\beta_i\beta_j - g_{0 0}} \\
                K_{ij} &= \frac{1}{2\alpha}\left(\nabla_i \beta_j + \nabla_j \beta_i\right) \\
                K &= \gamma^{ij} K_{ij}
            \end{align}
        """, ignore_warning=True)
        for i in range(3):
            for j in range(3):
                assert_equal(KDD[i][j], 0, suppress_message=True)

    def test_example_6_2(self):
        nl.parse_latex(r"""
            \begin{align}
                R_{ij} &= \partial_k \Gamma^k_{ij} - \partial_j \Gamma^k_{ik}
                    + \Gamma^k_{ij}\Gamma^l_{kl} - \Gamma^l_{ik}\Gamma^k_{lj} \\
                R &= \gamma^{ij} R_{ij} \\
                E &= \frac{1}{16\pi}\left(R + K^{{2}} - K_{ij}K^{ij}\right) \\
                p_i &= \frac{1}{8\pi}\left(D_j \gamma^{jk} K_{ki} - D_i K\right)
            \end{align}
        """, ignore_warning=True)
        # assert_equal(E, 0, suppress_message=True)
        self.assertEqual(sp.simplify(E), 0)
        for i in range(3):
            assert_equal(pD[i], 0, suppress_message=True)

    @staticmethod
    def test_metric_symmetry():
        parse_latex(r"""
            % vardef -zero gDD::3D
            g_{1 0} = 1 \\
            g_{2 0} = 2
            % assign -metric gDD
        """)
        assert_equal(gDD[0][1], 1, suppress_message=True)
        assert_equal(gDD[0][2], 2, suppress_message=True)
        parse_latex(r"""
            % vardef -zero gDD::3D
            g_{0 1} = 1 \\
            g_{0 2} = 2
            % assign -metric gDD
        """)
        assert_equal(gDD[1][0], 1, suppress_message=True)
        assert_equal(gDD[2][0], 2, suppress_message=True)

    @staticmethod
    def test_metric_inverse():
        for DIM in range(2, 5):
            parse_latex(r"""
                % vardef -metric gDD::{DIM}D
                % attrib index [a-c]::{DIM}D
                \Delta^a_c = g^{{ab}} g_{{bc}}
            """.format(DIM=DIM))
            for i in range(DIM):
                for j in range(DIM):
                    value = 1 if i == j else 0
                    assert_equal(DeltaUD[i][j], value, suppress_message=True)
        for DIM in range(2, 5):
            parse_latex(r"""
                % vardef -metric gUU::{DIM}D
                % attrib index [a-c]::{DIM}D
                \Delta^a_c = g^{{ab}} g_{{bc}}
            """.format(DIM=DIM))
            for i in range(DIM):
                for j in range(DIM):
                    value = 1 if i == j else 0
                    assert_equal(DeltaUD[i][j], value, suppress_message=True)

if __name__ == '__main__':
    unittest.main()
