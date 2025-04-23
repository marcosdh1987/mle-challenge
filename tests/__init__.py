# Allow tests directory to be recognized as a package
# (needed so tests.conftest can be imported by test modules)
__import__("sys")
__import__("os")