"""
Copyright 2013 Steven Diamond, Andrej Muhic

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.exp import exp


from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
import numpy as np
from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import exp_canon
from cvxpy.reductions.dcp2cone.atom_canonicalizers.power_canon import power_canon
from cvxpy.atoms.elementwise.power import power


def elu_canon(expr, args):
    shape = expr.shape
    x = args[0]
    t1 = Variable(shape)
    t2 = Variable(shape)
    x1 = Variable(shape, nonpos=True)
    x2 = Variable(shape, nonneg=True)

    exp_x1_expression = exp(x1)
    value_x1, value_constr_x1 = exp_canon(exp_x1_expression, exp_x1_expression.args)
    #TODO: add nontrivial power option
    id_x2_expression = power(x2, 1.0)
    value_x2, value_constr_x2 = power_canon(id_x2_expression, id_x2_expression.args)
    obj = t1 + t2
    constraints = value_constr_x1 + value_constr_x2 + [t1 >= value_x1, t2 >= value_x2, x == x1 + x2]
    return obj, constraints