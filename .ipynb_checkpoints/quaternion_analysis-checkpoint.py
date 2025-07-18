
from sympy import symbols, sin, cos, sinh, cosh, sqrt, log, atan2, acos, simplify, exp
from sympy.algebras.quaternion import Quaternion

def is_pure_imaginary(q):
    return q.a == 0

def norm(q):
    return sqrt(q.a**2 + q.b**2 + q.c**2 + q.d**2)

def unit_imag(q):
    """Вернёт мнимую единичную часть кватерниона"""
    n = sqrt(q.b**2 + q.c**2 + q.d**2)
    if n == 0:
        return Quaternion(0, 0, 0, 0)
    return Quaternion(0, q.b/n, q.c/n, q.d/n)

def exp_quaternion(q):
    """Экспонента кватерниона"""
    a = q.a
    v_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
    if v_norm == 0:
        return Quaternion(exp(a), 0, 0, 0)
    e_a = exp(a)
    s = sin(v_norm) / v_norm
    return Quaternion(e_a * cos(v_norm), *(e_a * s * x for x in (q.b, q.c, q.d)))

def sin_quaternion(q):
    """Синус кватерниона"""
    a, b, c, d = q.a, q.b, q.c, q.d
    v_norm = sqrt(b**2 + c**2 + d**2)
    if v_norm == 0:
        return Quaternion(sin(a), 0, 0, 0)
    u = unit_imag(q)
    sin_a = sin(a)
    cos_a = cos(a)
    sinh_v = sinh(v_norm)
    cosh_v = cosh(v_norm)
    real = sin_a * cosh_v
    imag_scale = cos_a * sinh_v
    return Quaternion(real,
                      imag_scale * u.b,
                      imag_scale * u.c,
                      imag_scale * u.d)

def cos_quaternion(q):
    """Косинус кватерниона"""
    a, b, c, d = q.a, q.b, q.c, q.d
    v_norm = sqrt(b**2 + c**2 + d**2)
    if v_norm == 0:
        return Quaternion(cos(a), 0, 0, 0)
    u = unit_imag(q)
    sin_a = sin(a)
    cos_a = cos(a)
    sinh_v = sinh(v_norm)
    cosh_v = cosh(v_norm)
    real = cos_a * cosh_v
    imag_scale = -sin_a * sinh_v
    return Quaternion(real,
                      imag_scale * u.b,
                      imag_scale * u.c,
                      imag_scale * u.d)

def log_quaternion(q):
    """Логарифм кватерниона"""
    q_norm = norm(q)
    v_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
    if v_norm == 0:
        return Quaternion(log(q_norm), 0, 0, 0)
    theta = acos(q.a / q_norm)
    u = unit_imag(q)
    return Quaternion(log(q_norm),
                      theta * u.b,
                      theta * u.c,
                      theta * u.d)
