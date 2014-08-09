import numpy

def Branin(x):
    """ This function is usually evaluated on the square x_1 \in [-5, 10], x_2 \in [0, 15]. Global minimum
    is at x = [-pi, 12.275], [pi, 2.275] and [9.42478, 2.475] with minima f(x*) = 0.397887.
    """
    a = 1
    b = 5.1 / (4 * pow(numpy.pi, 2.0))
    c = 5 / numpy.pi
    r = 6
    s = 10
    t = 1 / (8 * numpy.pi)
    return (a * pow(x[1] - b * pow(x[0], 2.0) + c * x[0] - r, 2.0) + s * (1-t) * numpy.cos(x[0]) + s)
