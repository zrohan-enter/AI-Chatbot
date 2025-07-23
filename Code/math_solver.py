# math_solver.py

import re
import sympy as sp
from typing import Optional

class MathSolver:
    """Handles detection and solving of mathematical queries, including natural language."""
    
    def __init__(self):
        self.math_keywords = [
            r'solve\s+.*',  # Broadly captures "solve" followed by any characters
            r'\\d+\\s*[\\+\\-\\*\\/\\^]\\s*\\d+',  # Basic arithmetic like "15 * 3"
            r'differentiate\\s+',  # Calculus
            r'integrate\\s+',  # Calculus
            r'simplify\\s+',  # Simplify expressions
            r'=\\s*\\d+',  # Equations with equals
            r'[a-zA-Z]\\s*\\^\\s*\\d+',  # Variables with powers
            r'(add|subtract|multiply|divide|times|plus|minus)\\s+[a-zA-Z0-9]+\\s+(and|plus|minus|times|by|divided by)\\s+[a-zA-9]+',
            r'what is\\s+.*\\?$', # Catches "what is" followed by a question
            r'calculate\\s+.*', # Catches "calculate" followed by an expression
            r'evalu\\w+\\s+.*', # Catches "evaluate" followed by an expression
        ]

    def is_math_query(self, user_input: str) -> bool:
        """Check if the input is a mathematical query."""
        return any(re.search(pattern, user_input.lower()) for pattern in self.math_keywords)

    def parse_natural_language_math(self, user_input: str) -> str:
        """Convert natural language math phrases to SymPy-compatible expressions."""
        user_input = user_input.lower().strip()
        word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        operation_map = {
            'add': '+', 'plus': '+', 'subtract': '-', 'minus': '-',
            'multiply': '*', 'times': '*', 'divide': '/', 'divided by': '/',
            'and': '+'  # 'by' is handled separately
        }
        for word, num in word_to_num.items():
            user_input = re.sub(r'\b' + word + r'\b', num, user_input)
        primary_op = None
        for word, symbol in operation_map.items():
            if re.search(r'\b' + word + r'\b', user_input):
                primary_op = symbol
                user_input = re.sub(r'\b' + word + r'\b', ' ', user_input)
                break
        user_input = re.sub(r'\b(by|and|plus|minus|times|divided by)\b', ' ', user_input)
        parts = re.split(r'\s+', user_input.strip())
        parts = [p for p in parts if p]
        if primary_op and len(parts) >= 2:
            final_expression = f"{parts[0]} {primary_op} {parts[1]}"
        else:
            final_expression = ' '.join(parts)
        return final_expression

    def solve_math(self, user_input: str) -> str:
        """Solve a mathematical expression."""
        try:
            user_input_lower = user_input.lower().strip()

            # Handle specific commands like 'differentiate', 'integrate', 'simplify' first
            if user_input_lower.startswith('differentiate'):
                expression = user_input_lower.replace('differentiate', '', 1).strip()
                expr = sp.sympify(expression)
                derivative = sp.diff(expr, sp.Symbol('x')) # Assuming 'x' as variable
                return f"Derivative: {derivative}"
            elif user_input_lower.startswith('integrate'):
                expression = user_input_lower.replace('integrate', '', 1).strip()
                expr = sp.sympify(expression)
                integral = sp.integrate(expr, sp.Symbol('x')) # Assuming 'x' as variable
                return f"Integral: {integral}"
            elif user_input_lower.startswith('simplify'):
                expression = user_input_lower.replace('simplify', '', 1).strip()
                expr = sp.sympify(expression)
                simplified = sp.simplify(expr)
                return f"Simplified: {simplified}"
            
            # This is the crucial part for "solve" and general expressions
            elif user_input_lower.startswith('solve '):
                # Remove "solve " prefix to get just the expression
                expression = user_input_lower[len('solve '):].strip()
                expr = sp.sympify(expression)
            else:
                # For inputs like just "2+2" or "x+y"
                expr = sp.sympify(user_input_lower)

            # Evaluate the expression
            if expr.is_number:
                result = expr.evalf()
                if result.is_integer:
                    return f"Result: {int(result)}"
                else:
                    return f"Result: {float(result):.4f}".rstrip('0').rstrip('.')
            else:
                # If it's not a direct number (e.g., an algebraic expression), simplify it
                result = sp.simplify(expr)
                return f"Result: {result}"

        except sp.SympifyError:
            return "Bot: Invalid mathematical expression. Please check your input."
        except Exception as e:
            return f"Bot: Error solving math problem: {str(e)}"