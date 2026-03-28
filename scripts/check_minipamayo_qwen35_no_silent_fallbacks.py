#!/usr/bin/env python3
"""Reject silent fallback patterns in minipamayo-qwen-3-5 pipeline code."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


SUSPICIOUS_BASE_NAMES = {
    "checkpoint",
    "checkpoint_args",
    "extract_summary",
    "frame",
    "metrics",
    "payload",
    "raw_config",
    "raw_extract",
    "raw_job",
    "record",
    "run_metadata",
    "source_frame",
    "stage1_metadata",
    "summary",
    "token_cfg",
    "quant_cfg",
}

ALLOWED_GET_DEFAULT_KEYS = {
    "path_base",
}


@dataclass
class Violation:
    path: Path
    lineno: int
    col: int
    message: str


def is_empty_container_literal(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    return False


def is_suspicious_name(name: str) -> bool:
    return (
        name in SUSPICIOUS_BASE_NAMES
        or name.endswith("_metadata")
        or name.endswith("_summary")
        or name.endswith("_record")
        or name.endswith("_payload")
        or name.endswith("_checkpoint")
    )


def expr_is_suspicious_source(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return is_suspicious_name(node.id)
    if isinstance(node, ast.Attribute):
        return expr_is_suspicious_source(node.value) or is_suspicious_name(node.attr)
    if isinstance(node, ast.Subscript):
        return expr_is_suspicious_source(node.value)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            return expr_is_suspicious_source(node.func.value)
        return False
    return False


def extract_get_default(node: ast.Call) -> ast.AST | None:
    if len(node.args) >= 2:
        return node.args[1]
    for keyword in node.keywords:
        if keyword.arg == "default":
            return keyword.value
    return None


def extract_get_key(node: ast.Call) -> str | None:
    if not node.args:
        return None
    key_node = node.args[0]
    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
        return key_node.value
    return None


def is_default_like(node: ast.AST) -> bool:
    if is_empty_container_literal(node):
        return False
    return isinstance(
        node,
        (
            ast.Constant,
            ast.Name,
            ast.Attribute,
            ast.Call,
            ast.Subscript,
            ast.Dict,
            ast.List,
            ast.Tuple,
            ast.Set,
        ),
    )


class SilentFallbackVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[Violation] = []
        self.function_stack: list[str] = []

    def add_violation(self, node: ast.AST, message: str) -> None:
        self.violations.append(
            Violation(
                path=self.path,
                lineno=getattr(node, "lineno", 1),
                col=getattr(node, "col_offset", 0) + 1,
                message=message,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Return(self, node: ast.Return) -> None:
        current_function = self.function_stack[-1] if self.function_stack else ""
        if current_function.startswith("require_"):
            if node.value is None or (
                isinstance(node.value, ast.Constant) and node.value.value is None
            ):
                self.add_violation(node, "`require_*` helpers must raise RuntimeError instead of returning None.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            default_node = extract_get_default(node)
            if default_node is not None and expr_is_suspicious_source(node.func.value):
                key = extract_get_key(node)
                if key not in ALLOWED_GET_DEFAULT_KEYS and not is_empty_container_literal(default_node):
                    self.add_violation(
                        node,
                        "Silent fallback via `.get(..., default)` is not allowed for canonical pipeline artifacts.",
                    )
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, ast.Or) and len(node.values) >= 2:
            left = node.values[0]
            if expr_is_suspicious_source(left):
                for fallback in node.values[1:]:
                    if is_default_like(fallback):
                        self.add_violation(
                            node,
                            "Silent fallback via `x or default` is not allowed for canonical pipeline artifacts.",
                        )
                        break
        self.generic_visit(node)


def check_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = SilentFallbackVisitor(path)
    visitor.visit(tree)
    return visitor.violations


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv[1:] if arg.endswith(".py")]
    violations: list[Violation] = []
    for path in paths:
        violations.extend(check_file(path))

    if not violations:
        return 0

    for violation in violations:
        print(
            f"{violation.path}:{violation.lineno}:{violation.col}: {violation.message}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
