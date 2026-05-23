"""
tools/calculator.py — Safe math expression evaluator using ast.literal_eval
and a restricted node visitor. No exec(), no eval() on arbitrary code.

Handles arithmetic, basic math functions, and unit conversions.
For complex computation, use python_repl instead.
"""

import ast
import math
import operator
from typing import Any

from tools.base import BaseTool

# Allowed operators
OPERATORS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math functions (no builtins that could be dangerous)
SAFE_FUNCTIONS = {
    "sqrt":  math.sqrt,
    "ceil":  math.ceil,
    "floor": math.floor,
    "abs":   abs,
    "round": round,
    "log":   math.log,
    "log2":  math.log2,
    "log10": math.log10,
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "exp":   math.exp,
    "pi":    math.pi,
    "e":     math.e,
    "inf":   math.inf,
    "factorial": math.factorial,
}


class _SafeEvaluator(ast.NodeVisitor):
    """AST visitor that only allows whitelisted nodes."""

    def visit(self, node: ast.AST) -> Any:
        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type not in OPERATORS:
            raise ValueError(f"Unary operator {op_type.__name__} not allowed.")
        operand = self.visit(node.operand)
        return OPERATORS[op_type](operand)

    def visit_Num(self, node):  # Python 3.7 compat
        return node.n

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Constant type {type(node.value)} not allowed.")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed.")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(
                f"Function '{func_name}' not allowed. "
                f"Allowed: {sorted(SAFE_FUNCTIONS)}"
            )
        func = SAFE_FUNCTIONS[func_name]
        args = [self.visit(a) for a in node.args]
        return func(*args)

    def visit_Name(self, node):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Name '{node.id}' not allowed.")

    def generic_visit(self, node):
        raise ValueError(f"AST node type {type(node).__name__} not allowed.")


def safe_eval(expression: str) -> Any:
    """Evaluate a math expression safely. Raises ValueError on invalid input."""
    tree = ast.parse(expression.strip(), mode="eval")
    evaluator = _SafeEvaluator()
    return evaluator.visit(tree)


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluates mathematical expressions safely. "
        "Supports: +, -, *, /, **, %, //, sqrt(), log(), sin(), cos(), "
        "tan(), abs(), round(), ceil(), floor(), factorial(), exp(), "
        "and constants pi and e. "
        "Use for quick arithmetic. For complex logic or loops, use python_repl."
    )

    def run(self, expression: str) -> str:
        try:
            result = safe_eval(expression)

            # Format nicely
            if isinstance(result, float):
                if result == int(result) and abs(result) < 1e15:
                    return str(int(result))
                return f"{result:.10g}"  # up to 10 sig figs, no trailing zeros
            return str(result)

        except ZeroDivisionError:
            return "Error: division by zero."
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error evaluating expression: {e}"

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": (
                                "A mathematical expression to evaluate. "
                                "Examples: '2 ** 10', 'sqrt(144)', "
                                "'log(100, 10)', '17 * 43 + 12'"
                            ),
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
