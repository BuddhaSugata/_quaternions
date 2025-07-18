from sympy.algebras.quaternion import Quaternion
from sympy import symbols, simplify, trigsimp, pretty, latex
from sympy import sinh, cosh, tanh, asinh, acosh, atanh
from sympy import sin, cos, tan, asin, acos, atan
from IPython.display import Math
from sympy.simplify.fu import fu
from quaternion_analysis import sin_quaternion, exp_quaternion, log_quaternion, norm

def quaternion_operator_matrix(f, q_sym=None):
    """
    Возвращает матрицу преобразования 4×4 в базисе (1, i, j, k),
    такую что: M * [a, b, c, d]^T = f(q)^T,
    где f: Quaternion → Quaternion.
    
    f: функция, принимающая Quaternion и возвращающая Quaternion
    q_sym: опциональный символический Quaternion
    """
    from sympy import symbols, Matrix
    from sympy.algebras.quaternion import Quaternion

    a, b, c, d = symbols('a b c d')
    q = Quaternion(a, b, c, d) if q_sym is None else q_sym
    q_result = f(q)

    # Вектор результата [a', b', c', d']^T
    result_vec = Matrix([
        q_result.a,
        q_result.b,
        q_result.c,
        q_result.d
    ])

    # Матрица преобразования M = ∂f/∂[a, b, c, d]^T
    M = result_vec.jacobian([a, b, c, d])

    return M

def quaternion_operator_matrix2(f, q_sym=None):
    """
    Возвращает матрицу преобразования 4×4 в базисе (1, i, j, k),
    эквивалентную применению операции `f` к произвольному кватерниону q = a + bi + cj + dk.
    
    f: функция от Quaternion → Quaternion
    """
    from sympy import symbols, Matrix
    from sympy.algebras.quaternion import Quaternion

    a, b, c, d = symbols('a b c d')
    q = Quaternion(a, b, c, d) if q_sym is None else q_sym
    q_result = f(q)

    # Результат в виде строки [a', b', c', d']
    result_vec = Matrix([q_result.a, q_result.b, q_result.c, q_result.d])

    # Матрица преобразования такая, что: [a b c d] * M = result_vec
    from sympy import Identity, solve
    base_vec = Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    M = result_vec.jacobian([a, b, c, d]).T  # каждый столбец — частные производные по a, b, c, d

    return M

a, b, c, d, theta, x, y, dTheta = symbols('a b c d theta x y theta_d')
q = Quaternion(a, b, c, d)
r = Quaternion(cos(theta/2),sin(theta/2),0,0)
r_ = Quaternion(cos(theta/2),sin(theta/2),0,0)
d = Quaternion(cos(theta/2),sin(theta/2),0,0)
ddot = Quaternion(-sin(theta/2),cos(theta/2),0,0) * dTheta
qdot = (d * q * ddot + ddot * q * d)/2
q_ = r * q * r_ 
simp = fu(trigsimp(simplify(q_)))
simp2 = fu(trigsimp(simplify(qdot)))
latex_string = latex(quaternion_operator_matrix(lambda q: simp))
display(Math(latex_string))
# display(Math(latex(sin_quaternion(r))))
print(qdot)
latex_string2 = latex(quaternion_operator_matrix(lambda q: simp2))
display(Math(latex_string2))

d = Quaternion(2**(-0.5), 2**(-0.5),0, 0)
r1 = Quaternion(2**(-0.5), 0, 2**(-0.5), 0)
r2 = Quaternion(2**(-0.5), 0, 0, 2**(-0.5))
q__ = d*q*d
q___1 = r1 * d*q*d * r1
q___2 = r2 * d*q*d * r2
print(q__)
print(q___1)
print(q___2)
simp3 = fu(trigsimp(simplify(q__)))
latex_string3 = latex(quaternion_operator_matrix(lambda q: simp3))
display(Math(latex_string3))
simp4 = fu(trigsimp(simplify(q___1)))
latex_string4 = latex(quaternion_operator_matrix(lambda q: simp4))
display(Math(latex_string4))