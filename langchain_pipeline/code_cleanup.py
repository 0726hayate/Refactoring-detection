"""
Code cleanup utilities for reducing token usage.

Strips comments, docstrings, extra whitespace, and blank lines from code
before passing it to the LLM. Copied from compress_code.py (standalone).
"""
import re


def minify_python(code: str) -> str:
    """Make Python source smaller (fewer tokens) before sending to the LLM."""
    if not code or not code.strip():
        return code
    try:
        import python_minifier
        return python_minifier.minify(
            code,
            rename_locals=False,
            rename_globals=False,
            remove_literal_statements=True,
            hoist_literals=False,
            remove_annotations=True,
            remove_pass=False,
            remove_object_base=True,
            combine_imports=True,
            remove_explicit_return_none=True,
        )
    except Exception:
        return basic_cleanup(code)


def basic_cleanup(code: str) -> str:
    """Simple fallback cleanup — works on any text, no imports needed."""
    if not code or not code.strip():
        return code

    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)

    lines = code.split('\n')
    result = []
    for line in lines:
        in_string = False
        string_char = None
        comment_start = None
        i = 0
        while i < len(line):
            c = line[i]
            if in_string:
                if c == '\\':
                    i += 2
                    continue

                if c == string_char:
                    in_string = False
            else:
                if c in ('"', "'"):
                    in_string = True
                    string_char = c

                elif c == '#':
                    comment_start = i
                    break
            i += 1

        if comment_start is not None:
            line = line[:comment_start].rstrip()

        result.append(line.rstrip())

    text = '\n'.join(result)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def minify_java(code: str) -> str:
    """Shrink Java source by removing comments and extra whitespace."""
    if not code or not code.strip():
        return code

    code = re.sub(r'/\*[\s\S]*?\*/', '', code)

    code = re.sub(r'//[^\n]*', '', code)

    code = re.sub(r'\n{3,}', '\n\n', code)

    lines = [line.rstrip() for line in code.split('\n')]
    return '\n'.join(lines).strip()
