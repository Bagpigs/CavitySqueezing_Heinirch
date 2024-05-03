from numbers import Number

import sympy
from sympy import simplify, solve, Basic, Pow, Mul, Function, preorder_traversal, Eq, exp, I, sqrt, Matrix, sin, cos
from sympy.physics.quantum import Commutator, Operator, Dagger

# StackOverflow start
from sympy.core.operations import AssocOp


def apply_ccr(expr, ccr, reverse=False):
    if not isinstance(expr, Basic):
        raise TypeError("The expression to simplify is not a sympy expression.")

    if not isinstance(ccr, Eq):
        if isinstance(ccr, Basic):
            ccr = Eq(ccr, 0)
        else:
            raise TypeError("The canonical commutation relation is not a sympy expression.")

    comm = None

    for node in preorder_traversal(ccr):
        if isinstance(node, Commutator):
            comm = node
            break

    if comm is None:
        raise ValueError("The cannonical commutation relation doesn not include a commutator.")

    solutions = solve(ccr, comm)

    if len(solutions) != 1:
        raise ValueError("There are more solutions to the cannonical commutation relation.")

    value = solutions[0]

    A = comm.args[0]
    B = comm.args[1]

    if reverse:
        (A, B) = (B, A)
        value = -value

    def is_expandable_pow_of(base, expr):
        return isinstance(expr, Pow) \
               and base == expr.args[0] \
               and isinstance(expr.args[1], Number) \
               and expr.args[1] >= 1

    def walk_tree(expr):
        if isinstance(expr, Number):
            return expr

        if not isinstance(expr, AssocOp) and not isinstance(expr, Function):
            return expr.copy()

        elif not isinstance(expr, Mul):
            return expr.func(*(walk_tree(node) for node in expr.args))

        else:
            args = [arg for arg in expr.args]

            for i in range(len(args) - 1):
                x = args[i]
                y = args[i + 1]

                if B == x and A == y:
                    args = args[0:i] + [A * B - value] + args[i + 2:]
                    return walk_tree(Mul(*args).expand())

                if B == x and is_expandable_pow_of(A, y):
                    ypow = Pow(A, y.args[1] - 1)
                    args = args[0:i] + [A * B - value, ypow] + args[i + 2:]
                    return walk_tree(Mul(*args).expand())

                if is_expandable_pow_of(B, x) and A == y:
                    xpow = Pow(B, x.args[1] - 1)
                    args = args[0:i] + [xpow, A * B - value] + args[i + 2:]
                    return walk_tree(Mul(*args).expand())

                if is_expandable_pow_of(B, x) and is_expandable_pow_of(A, y):
                    xpow = Pow(B, x.args[1] - 1)
                    ypow = Pow(A, y.args[1] - 1)
                    args = args[0:i] + [xpow, A * B - value, ypow] + args[i + 2:]
                    return walk_tree(Mul(*args).expand())

            return expr.copy()

    return walk_tree(expr)


Basic.apply_ccr = lambda self, ccr, reverse=False: apply_ccr(self, ccr, reverse)
# StackOverflow end


# Start set6
phi_a, phi_b = sympy.symbols('phi_a,phi_b', real=True)
a = Operator('a')
b = Operator('b')
a_prime = Operator('a_prime')
b_prime = Operator('b_prime')

bs_1 = 1/ sqrt(2)* Matrix([[1, -I], [-I, 1]])
phase = Matrix([[exp(I * phi_a), 0], [0, exp(I * phi_b)]])
bs_2 = 1/ sqrt(2)* Matrix([[1, I], [I, 1]])
optics = bs_2 * phase * bs_1
op_vec = Matrix([a,b])
optics_out = optics * op_vec
print(optics_out[0].rewrite(sin).rewrite(cos).simplify().simplify())
print(optics_out[1].simplify())

a_prime = 1 / 2 * (exp(I * phi_a) * a - I * exp(I * phi_a) * b + exp(I * phi_b) * a + I * exp(I * phi_b) * b)
b_prime = 1 / 2 * (I * exp(I * phi_a) * a + exp(I * phi_a) * b - exp(I * phi_b) * I * a + exp(I * phi_b) * b)
# print(a_prime)
J_z_prime = 1/2 * (Dagger(a_prime)* a_prime - Dagger(b_prime) * b_prime)
# print(J_z_prime.simplify())
# print(J_z_prime.rewrite(exp).simplify())


# If I write Dagger(phi_b) =  5 * I , and then solve for phi_b, I get [].
# If I write Dagger(phi_b) =  5  , and then solve for phi_b, I get [5].
# If I write phi_b =  5  , and then solve for Dagger(phi_b), I get [] (This is because I cant even solve for 2* phi_b. I just can solve for pure variables, I guess.-

# eq1 = Eq(phi_b, 5)
# print('hi',sympy.solve(eq1,phi_b))

# eq = sympy.Eq(a_prime+b_prime,10)
ccr = sympy.Eq(Commutator(a, Dagger(a)), 1)

# x = (b * Dagger(b) - Dagger(b) * b)
# x1 = J_z_prime.expand().apply_ccr(ccr)
# x2 = x1.expand().apply_ccr(ccr)
# x3 = simplify(x2)
# x4 = x3.expand().apply_ccr(ccr)
# print(x4)
# print(x)
# print(simplify(x))
# # print(sympy.solve(com,a*Dagger(a)))
# print(sympy.solve(eq,a_prime))