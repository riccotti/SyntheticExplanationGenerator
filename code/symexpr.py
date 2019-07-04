import numpy as np
from sympy import sympify, Symbol, re


# def generate_expression(scope, num_operations=10, p_binary=0.7, p_parenthesis=0.3):
#     unaries = ['sqrt(%s)', 'exp(%s)', 'log(%s)', 'sin(%s)', 'cos(%s)', 'tan(%s)',
#                'sinh(%s)', 'cosh(%s)', 'tanh(%s)', 'asin(%s)', 'acos(%s)',
#                'atan(%s)', '-%s', 'sign(%s)']
#     binaries = ['%s+%s', '%s-%s', '%s*%s', '%s/%s', '%s**%s']
#
#     scope = list(scope)  # make a copy first, append as we go
#     for _ in range(num_operations):
#         if np.random.random() < p_binary:  # decide unary or binary operator
#             ex = np.random.choice(binaries) % (np.random.choice(scope), np.random.choice(scope))
#             if np.random.random() < p_parenthesis:
#                 ex = '(%s)' % ex
#             scope.append(ex)
#         else:
#             scope.append(np.random.choice(unaries) % np.random.choice(scope))
#     return scope[-1]  # return most recent expressions


def generate_expression(scope, p_binary=0.7, p_parenthesis=0.3):
    unaries = ['sqrt(%s)', 'exp(%s)', 'log(%s)', 'sin(%s)', 'cos(%s)', 'tan(%s)',
               'sinh(%s)', 'cosh(%s)', 'tanh(%s)', 'asin(%s)', 'acos(%s)',
               'atan(%s)', '-%s', 'sign(%s)', '%s']
    binaries = ['%s+%s', '%s-%s', '%s*%s', '%s/%s', '%s**%s']

    scope = list(scope)  # make a copy first, append as we go
    expr = list()
    for variable in scope:
        if np.random.random() < p_binary:  # decide unary or binary operator
            other = np.random.choice(list(set(scope) - {variable}))
            if np.random.random() < 0.5:
                term = np.random.choice(binaries) % (variable, other)
            else:
                term = np.random.choice(binaries) % (other, variable)
            if np.random.random() < p_parenthesis:
                term = '(%s)' % term
        else:
            term = np.random.choice(unaries) % variable
        expr.append(term)

    expr = '+'.join(expr)

    return expr


def symbolize(s):
    """
    Converts a a string (equation) to a SymPy symbol object
    """

    s1 = s.replace('.', '*')
    s2 = s1.replace('^', '**')
    s3 = sympify(s2)

    return (s3)


def eval_multinomial(s, vals=None, symbolic_eval=False):
    """
    Evaluates polynomial at vals.
    vals can be simple list, dictionary, or tuple of values.
    vals can also contain symbols instead of real values provided those symbols have been declared before using SymPy
    """
    from sympy import Symbol
    sym_s = symbolize(s)
    sym_set = sym_s.atoms(Symbol)
    sym_lst = []
    for s in sym_set:
        sym_lst.append(str(s))
    sym_lst.sort()
    if symbolic_eval is False and len(sym_set) != len(vals):
        print("Length of the input values did not match number of variables and symbolic evaluation is not selected")
        return None
    else:
        if type(vals) == list:
            sub = list(zip(sym_lst, vals))
        elif type(vals) == dict:
            l = list(vals.keys())
            l.sort()
            lst = []
            for i in l:
                lst.append(vals[i])
            sub = list(zip(sym_lst, lst))
        elif type(vals) == tuple:
            sub = list(zip(sym_lst, list(vals)))
        result = sym_s.subs(sub)

    return result


def flip(y, p):
    import numpy as np
    lst = []
    for i in range(len(y)):
        f = np.random.choice([1, 0], p=[p, 1 - p])
        lst.append(f)
    lst = np.array(lst)
    return np.array(np.logical_xor(y, lst), dtype=int)


def gen_classification_symbolic(expr, n_samples=100, flip_y=0.0):
    """
    Generates classification sample based on a symbolic expression.
    Calculates the output of the symbolic expression at randomly generated (Gaussian distribution) points and
    assigns binary classification based on sign.
    m: The symbolic expression. Needs x1, x2, etc as variables and regular python arithmatic symbols to be used.
    n_samples: Number of samples to be generated
    n_features: Number of variables. This is automatically inferred from the symbolic expression. So this is ignored
                in case a symbolic expression is supplied. However if no symbolic expression is supplied then a
                default simple polynomial can be invoked to generate classification samples with n_features.
    flip_y: Probability of flipping the classification labels randomly. A higher value introduces more noise and make
            the classification problem harder.
    Returns a numpy ndarray with dimension (n_samples,n_features+1). Last column is the response vector.
    """

    sym_m = sympify(expr)
    n_features = len(sym_m.atoms(Symbol))
    evals = []
    lst_features = []
    for i in range(n_features):
        lst_features.append(np.random.normal(scale=5, size=n_samples))
    lst_features = np.array(lst_features)
    lst_features = lst_features.T
    for i in range(n_samples):
        evals.append(eval_multinomial(expr, vals=list(lst_features[i])))

    evals = np.array(evals)
    evals = np.array([re(e) for e in evals])
    evals_binary = evals > 0
    evals_binary = evals_binary.flatten()
    evals_binary = np.array(evals_binary, dtype=int)
    evals_binary = flip(evals_binary, p=flip_y)
    # evals_binary = evals_binary.reshape(n_samples, 1)

    # lst_features = lst_features.reshape(n_samples, n_features)
    # X = np.hstack((lst_features, evals_binary))
    X = lst_features.reshape(n_samples, n_features)
    Y = evals_binary
    Y1 = evals

    return X, Y, Y1