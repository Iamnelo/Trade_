"""Research-only tools. NOT importable from any strategy or live code path.

Anything in this subpackage may consult future data ("look-ahead") because it
is meant strictly for evaluation and diagnostics, never for signal generation
or trading. The Oracle benchmark lives here for exactly this reason.
"""
