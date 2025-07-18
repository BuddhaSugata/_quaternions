from sympy import symbols, sin, cos, sinh, cosh, I, re, im, simplify, expand

# z — произвольная комплексная переменная
z = symbols('z')

def Synh(z):
    """
    Синтетический гиперболический синус:
    Synh(z) = sinh(Im(z)) * cos(Re(z)) + i * cosh(Im(z)) * sin(Re(z))
    """
    theta = re(z)
    sigma = im(z)
    return sinh(sigma) * cos(theta) + I * cosh(sigma) * sin(theta)

def Cosh(z):
    """
    Синтетический гиперболический косинус:
    Cosh(z) = sinh(Im(z)) * sin(Re(z)) + i * cosh(Im(z)) * cos(Re(z))
    """
    theta = re(z)
    sigma = im(z)
    return sinh(sigma) * sin(theta) + I * cosh(sigma) * cos(theta)
