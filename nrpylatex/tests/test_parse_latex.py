""" parse_latex.py Unit Testing """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

# pylint: disable = import-error, protected-access, exec-used
from nrpylatex.assert_equal import assert_equal
from nrpylatex.parse_latex import *
from sympy import Function, Symbol, Matrix, simplify
import unittest, sys

class TestParser(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        ignore_override(True)

    def test_expression_1(self):
        expr = r'-(\frac{2}{3} + 2\sqrt[5]{x + 3})'
        self.assertEqual(
            str(parse_expr(expr)),
            '-2*(x + 3)**(1/5) - 2/3'
        )

    def test_expression_2(self):
        expr = r'e^{{\ln x}} + \sin(\sin^{-1} y) - \tanh(xy)'
        self.assertEqual(
            str(parse_expr(expr)),
            'x + y - tanh(x*y)'
        )

    def test_expression_3(self):
        expr = r'\partial_x (x^{{2}} + 2x)'
        self.assertEqual(
            str(parse_expr(expr).doit()),
            '2*x + 2'
        )

    def test_expression_4(self):
        function = Function('Tensor')(Symbol('T'))
        self.assertEqual(
            Parser._generate_covdrv(function, 'beta'),
            r'\nabla_\beta T = \partial_\beta T'
        )
        function = Function('Tensor')(Symbol('TUU'), Symbol('mu'), Symbol('nu'))
        self.assertEqual(
            Parser._generate_covdrv(function, 'beta'),
            r'\nabla_\beta T^{\mu \nu} = \partial_\beta T^{\mu \nu} + \text{Gamma}^\mu_{a \beta} (T^{a \nu}) + \text{Gamma}^\nu_{a \beta} (T^{\mu a})'
        )
        function = Function('Tensor')(Symbol('TUD'), Symbol('mu'), Symbol('nu'))
        self.assertEqual(
            Parser._generate_covdrv(function, 'beta'),
            r'\nabla_\beta T^\mu_\nu = \partial_\beta T^\mu_\nu + \text{Gamma}^\mu_{a \beta} (T^a_\nu) - \text{Gamma}^a_{\nu \beta} (T^\mu_a)'
        )
        function = Function('Tensor')(Symbol('TDD'), Symbol('mu'), Symbol('nu'))
        self.assertEqual(
            Parser._generate_covdrv(function, 'beta'),
            r'\nabla_\beta T_{\mu \nu} = \partial_\beta T_{\mu \nu} - \text{Gamma}^a_{\mu \beta} (T_{a \nu}) - \text{Gamma}^a_{\nu \beta} (T_{\mu a})'
        )

    def test_expression_5(self):
        delete_namespace()
        parse(r"""
            % vardef -diff_type=dD -metric 'gDD' (4D)
            % vardef -diff_type=dD 'vU' (4D)
            T^\mu_b = \nabla_b v^\mu
        """)
        function = Function('Tensor')(Symbol('vU_cdD'), Symbol('mu'), Symbol('b'))
        self.assertEqual(
            Parser._generate_covdrv(function, 'a'),
            r'\nabla_a \nabla_b v^\mu = \partial_a \nabla_b v^\mu + \text{Gamma}^\mu_{c a} (\nabla_b v^c) - \text{Gamma}^c_{b a} (\nabla_c v^\mu)'
        )

    def test_expression_6(self):
        function = Function('Tensor')(Symbol('g'))
        self.assertEqual(
            Parser._generate_liedrv(function, 'beta', 2),
            r'\mathcal{L}_\text{beta} g = \text{beta}^a \partial_a g + (2)(\partial_a \text{beta}^a) g'
        )
        function = Function('Tensor')(Symbol('gUU'), Symbol('i'), Symbol('j'))
        self.assertEqual(
            Parser._generate_liedrv(function, 'beta'),
            r'\mathcal{L}_\text{beta} g^{i j} = \text{beta}^a \partial_a g^{i j} - (\partial_a \text{beta}^i) g^{a j} - (\partial_a \text{beta}^j) g^{i a}'
        )
        function = Function('Tensor')(Symbol('gUD'), Symbol('i'), Symbol('j'))
        self.assertEqual(
            Parser._generate_liedrv(function, 'beta'),
            r'\mathcal{L}_\text{beta} g^i_j = \text{beta}^a \partial_a g^i_j - (\partial_a \text{beta}^i) g^a_j + (\partial_j \text{beta}^a) g^i_a'
        )
        function = Function('Tensor')(Symbol('gDD'), Symbol('i'), Symbol('j'))
        self.assertEqual(
            Parser._generate_liedrv(function, 'beta'),
            r'\mathcal{L}_\text{beta} g_{i j} = \text{beta}^a \partial_a g_{i j} + (\partial_i \text{beta}^a) g_{a j} + (\partial_j \text{beta}^a) g_{i a}'
        )

    def test_srepl_macro(self):
        delete_namespace()
        parse(r"""
            % srepl -persist "<1>'" -> "\text{<1>prime}"
            % srepl -persist "\text{<1..>}_<2>" -> "\text{(<1..>)<2>}"
            % srepl -persist "<1>_{<2>}" -> "<1>_<2>", "<1>_<2>" -> "\text{<1>_<2>}"
            % srepl -persist "\text{(<1..>)<2>}" -> "\text{<1..>_<2>}"
            % srepl -persist "<1>^{<2>}" -> "<1>^<2>", "<1>^<2>" -> "<1>^{{<2>}}"
        """)
        expr = r"x_n^4 + x'_n \exp(x_n y_n^2)"
        self.assertEqual(
            str(parse_expr(expr)),
            "x_n**4 + xprime_n*exp(x_n*y_n**2)"
        )
        delete_namespace()
        parse(r""" % srepl -persist "<1>'^{<2..>}" -> "\text{<1>prime}" """)
        expr = r"v'^{label}"
        self.assertEqual(
            str(parse_expr(expr)),
            "vprime"
        )

    def test_assignment_1(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef -diff_type=dD 'vU' (2D), 'wU' (2D)
                % keydef index [a-z] (2D)
                T^{ab}_c = \partial_c (v^a w^b)
            """)),
            {'vU', 'wU', 'vU_dD', 'wU_dD', 'TUUD'}
        )
        self.assertEqual(str(TUUD[0][0][0]),
            'vU0*wU_dD00 + vU_dD00*wU0'
        )

    def test_assignment_2(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef -const 'w'
                % vardef -diff_type=dD 'vU' (2D)
                % keydef index [a-z] (2D)
                T^a_c = \partial_c (v^a w)
            """)),
            {'vU', 'vU_dD', 'TUD'}
        )
        self.assertEqual(str(TUD),
            '[[vU_dD00*w, vU_dD01*w], [vU_dD10*w, vU_dD11*w]]'
        )

    def test_assignment_3(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef -diff_type=dD -metric 'gDD' (4D)
                % vardef -diff_type=dD 'vU' (4D)
                % keydef index [a-z] (4D)
                T^{ab} = \nabla^b v^a
            """)),
            {'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'vU', 'vU_dD', 'gDD_dD', 'GammaUDD', 'vU_cdD', 'vU_cdU', 'TUU'}
        )

    def test_assignment_4(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % keydef basis [x, y]
                % vardef 'uD' (2D), 'wD' (2D)
                % keydef index [a-z] (2D)
                u_x = x^{{2}} + 2x \\
                u_y = y\sqrt{x} \\
                v_a = u_a + w_a \\
                % assign -diff_type=dD 'wD', 'vD'
                T_{ab} = \partial^2_x v_x (\partial_b v_a)
            """)),
            {'uD', 'wD', 'vD', 'vD_dD', 'wD_dD', 'TDD'}
        )
        self.assertEqual(str(TDD),
            '[[2*wD_dD00 + 4*x + 4, 2*wD_dD01], [2*wD_dD10 + y/sqrt(x), 2*wD_dD11 + 2*sqrt(x)]]'
        )

    def test_assignment_5(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % keydef basis [x, y]
                % vardef 'uD' (2D), 'wD' (2D)
                % assign -diff_type=symbolic 'uD'
                % keydef index [a-z] (2D)
                u_x = x^{{2}} + 2x \\
                u_y = y\sqrt{x} \\
                v_a = u_a + w_a \\
                T_{bc} = \partial^2_x v_x (\vphantom{dD} \partial_c v_b)
            """)),
            {'uD', 'wD', 'vD', 'vD_dD', 'wD_dD', 'TDD'}
        )
        self.assertEqual(str(TDD),
            '[[2*wD_dD00 + 4*x + 4, 2*wD_dD01], [2*wD_dD10 + y/sqrt(x), 2*wD_dD11 + 2*sqrt(x)]]'
        )

    def test_assignment_6(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                    % vardef 'vD' (2D), 'uD' (2D), 'wD' (2D)
                    % keydef index [a-z] (2D)
                    T_{abc} = \vphantom{dD} ((v_a + u_a)_{,b} - w_{a,b})_{,c}
            """)),
            {'vD', 'uD', 'wD', 'TDDD', 'uD_dD', 'vD_dD', 'wD_dD', 'wD_dDD', 'uD_dDD', 'vD_dDD'}
        )
        self.assertEqual(str(TDDD[0][0][0]),
            'uD_dDD000 + vD_dDD000 - wD_dDD000'
        )

    def test_assignment_7(self):
        delete_namespace()
        parse(r"""
            % keydef basis [\theta, \phi]
            % vardef -zero 'gDD' (2D)
            % vardef -const 'r'
            % keydef index [a-z] (2D)
            \begin{align*}
                g_{0 0} &= r^{{2}} \\
                g_{1 1} &= r^{{2}} \sin^2(\theta)
            \end{align*}
            % assign -metric 'gDD'
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

    def test_assignment_8(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef -metric 'gDD' (4D)
                \gamma_{ij} = g_{ij}
            """)),
            {'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'gammaDD'}
        )
        self.assertEqual(str(gammaDD),
            '[[gDD11, gDD12, gDD13], [gDD12, gDD22, gDD23], [gDD13, gDD23, gDD33]]'
        )

    def test_assignment_9(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef 'TUU' (3D)
                % vardef 'vD' (2D)
                % keydef index i (2D)
                w^\mu = T^{\mu i} v_i
            """)),
            {'TUU', 'vD', 'wU'}
        )
        self.assertEqual(str(wU),
            '[TUU01*vD0 + TUU02*vD1, TUU11*vD0 + TUU12*vD1, TUU21*vD0 + TUU22*vD1]'
        )

    def test_assignment_10(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef -metric 'gDD'
                % vardef 'ADDD', 'AUUU'
                B^{a b}_c = A^{a b}_c
            """)),
            {'ADDD', 'BUUD', 'gdet', 'gDD', 'AUUD', 'AUUU', 'gUU', 'epsilonUUU'}
        )

    def test_example_1(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef 'hUD' (4D)
                h = h^\mu{}_\mu
            """)),
            {'hUD', 'h'}
        )
        self.assertEqual(str(h),
            'hUD00 + hUD11 + hUD22 + hUD33'
        )

    def test_example_2(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef -metric 'gUU' (3D)
                % vardef 'vD' (3D)
                v^\mu = g^{\mu\nu} v_\nu
            """)),
            {'gDD', 'gdet', 'epsilonDDD', 'gUU', 'vD', 'vU'}
        )
        self.assertEqual(str(vU),
            '[gUU00*vD0 + gUU01*vD1 + gUU02*vD2, gUU01*vD0 + gUU11*vD1 + gUU12*vD2, gUU02*vD0 + gUU12*vD1 + gUU22*vD2]'
        )

    def test_example_3(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef 'vU' (3D), 'wU' (3D)
                u_i = \epsilon_{ijk} v^j w^k
            """)),
            {'epsilonDDD', 'vU', 'wU', 'uD'}
        )
        self.assertEqual(str(uD),
            '[vU1*wU2 - vU2*wU1, -vU0*wU2 + vU2*wU0, vU0*wU1 - vU1*wU0]'
        )

    def test_example_4(self):
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef -diff_type=dD -symmetry=anti01 'FUU' (4D)
                % vardef -diff_type=dD -metric 'gDD' (4D)
                % vardef -const 'k'
                J^\mu = (4\pi k)^{-1} F^{\mu\nu}_{;\nu}
            """)),
            {'FUU', 'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'FUU_dD', 'gDD_dD', 'GammaUDD', 'FUU_cdD', 'JU'}
        )
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef -diff_type=dD -symmetry=anti01 'FUU' (4D)
                % vardef -diff_type=dD -metric 'gDD' (4D)
                % vardef -const 'k'
                J^\mu = (4\pi k)^{-1} \nabla_\nu F^{\mu\nu}
            """)),
            {'FUU', 'gUU', 'gdet', 'epsilonUUUU', 'gDD', 'FUU_dD', 'gDD_dD', 'GammaUDD', 'FUU_cdD', 'JU'}
        )
        delete_namespace()
        self.assertEqual(
            set(parse(r"""
                % vardef -diff_type=dD -symmetry=anti01 'FUU' (4D)
                % vardef -diff_type=dD -metric 'ghatDD' (4D)
                % vardef -const 'k'
                J^\mu = (4\pi k)^{-1} \hat{\nabla}_\nu F^{\mu\nu}
            """)),
            {'FUU', 'ghatUU', 'ghatdet', 'epsilonUUUU',  'ghatDD', 'FUU_dD', 'ghatDD_dD', 'GammahatUDD', 'FUU_cdhatD', 'JU'}
        )

    def test_example_5_1(self):
        delete_namespace()
        parse(r"""
            % keydef basis [t, r, \theta, \phi]
            % vardef -zero 'gDD' (4D)
            % vardef -const 'G', 'M'
            \begin{align}
                g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
                g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
                g_{\theta \theta} &= r^{{2}} \\
                g_{\phi \phi} &= r^{{2}} \sin^2\theta
            \end{align}
            % assign -metric 'gDD'
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
        self.assertEqual(str(gdet),
            'r**4*(2*G*M/r - 1)*sin(theta)**2/(-2*G*M/r + 1)'
        )

    def test_example_5_2(self):
        parse(r"""
            \begin{align}
                R^\alpha{}_{\beta\mu\nu} &= \partial_\mu \Gamma^\alpha_{\beta\nu} - \partial_\nu \Gamma^\alpha_{\beta\mu} + \Gamma^\alpha_{\mu\gamma}\Gamma^\gamma_{\beta\nu} - \Gamma^\alpha_{\nu\sigma}\Gamma^\sigma_{\beta\mu} \\
                K &= R^{\alpha\beta\mu\nu} R_{\alpha\beta\mu\nu} \\
                R_{\beta\nu} &= R^\alpha_{\beta\alpha\nu} \\
                R &= g^{\beta\nu} R_{\beta\nu} \\
                G_{\beta\nu} &= R_{\beta\nu} - \frac{1}{2}g_{\beta\nu}R
            \end{align}
        """)
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
        self.assertEqual(str(simplify(K)),
            '48*G**2*M**2/r**6'
        )
        assert_equal(R, 0, suppress_message=True)
        for i in range(3):
            for j in range(3):
                assert_equal(GDD[i][j], 0, suppress_message=True)

    @staticmethod
    def test_example_6_1():
        parse(r"""
            % keydef basis [r, \theta, \phi]
            \begin{align}
                \gamma_{ij} &= g_{ij} \\
                % assign -metric 'gammaDD'
                \beta_i &= g_{0 i} \\
                \alpha &= \sqrt{\gamma^{ij}\beta_i\beta_j - g_{0 0}} \\
                K_{ij} &= \frac{1}{2\alpha}\left(\nabla_i \beta_j + \nabla_j \beta_i\right) \\
                K &= \gamma^{ij} K_{ij}
            \end{align}
        """)
        for i in range(3):
            for j in range(3):
                assert_equal(KDD[i][j], 0, suppress_message=True)

    def test_example_6_2(self):
        parse(r"""
            \begin{align}
                R_{ij} &= \partial_k \Gamma^k_{ij} - \partial_j \Gamma^k_{ik}
                    + \Gamma^k_{ij}\Gamma^l_{kl} - \Gamma^l_{ik}\Gamma^k_{lj} \\
                R &= \gamma^{ij} R_{ij} \\
                E &= \frac{1}{16\pi}\left(R + K^{{2}} - K_{ij}K^{ij}\right) \\
                p_i &= \frac{1}{8\pi}\left(D_j \gamma^{jk} K_{ki} - D_i K\right)
            \end{align}
        """)
        # assert_equal(E, 0, suppress_message=True)
        self.assertEqual(simplify(E), 0)
        for i in range(3):
            assert_equal(pD[i], 0, suppress_message=True)

    @staticmethod
    def test_metric_symmetry():
        delete_namespace()
        parse(r"""
            % vardef -zero 'gDD'
            g_{1 0} = 1 \\
            g_{2 0} = 2
            % assign -metric 'gDD'
        """)
        assert_equal(gDD[0][1], 1, suppress_message=True)
        assert_equal(gDD[0][2], 2, suppress_message=True)
        delete_namespace()
        parse(r"""
            % vardef -zero 'gDD'
            g_{0 1} = 1 \\
            g_{0 2} = 2
            % assign -metric 'gDD'
        """)
        assert_equal(gDD[1][0], 1, suppress_message=True)
        assert_equal(gDD[2][0], 2, suppress_message=True)

    @staticmethod
    def test_metric_inverse():
        for DIM in range(2, 5):
            delete_namespace()
            parse(r"""
                % vardef -metric 'gDD' ({DIM}D)
                \Delta^a_c = g^{{ab}} g_{{bc}}
            """.format(DIM=DIM))
            for i in range(DIM):
                for j in range(DIM):
                    value = 1 if i == j else 0
                    assert_equal(DeltaUD[i][j], value, suppress_message=True)
        for DIM in range(2, 5):
            delete_namespace()
            parse(r"""
                % vardef -metric 'gUU' ({DIM}D)
                \Delta^a_c = g^{{ab}} g_{{bc}}
            """.format(DIM=DIM))
            for i in range(DIM):
                for j in range(DIM):
                    value = 1 if i == j else 0
                    assert_equal(DeltaUD[i][j], value, suppress_message=True)

if __name__ == '__main__':
    unittest.main()
