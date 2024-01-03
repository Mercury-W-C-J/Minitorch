from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    left_vals = list(vals)
    left_vals[arg] = vals[arg] - epsilon
    right_vals = list(vals)
    right_vals[arg] = vals[arg] + epsilon
    return (f(*right_vals) - f(*left_vals)) / (2 * epsilon)
    raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited = set()
    output = []

    def visit(visited, node):
        if node.unique_id not in visited:
            if not (node.is_constant() or node.is_leaf()):
                for iteration in node.parents:
                    visit(visited, iteration)
            visited.add(node.unique_id)
            output.insert(0, node)

    visit(visited, variable)
    return output

    raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    computation_sort = topological_sort(variable)
    deriv_dict = {variable.unique_id: deriv}
    for iterate in computation_sort:
        if iterate.is_leaf():
            iterate.accumulate_derivative(deriv_dict[iterate.unique_id])
        elif iterate.is_constant():
            continue
        else:
            for down_variable, down_deriv in iterate.chain_rule(
                deriv_dict[iterate.unique_id]
            ):
                if down_variable.unique_id in deriv_dict.keys():
                    deriv_dict[down_variable.unique_id] += down_deriv
                else:
                    deriv_dict[down_variable.unique_id] = down_deriv

    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
