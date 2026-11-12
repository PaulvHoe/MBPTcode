"""docs/warnings.md names every numerical flag and every warning class in src.

A flag is a name a solver raises when it stops short; a reader looks it up in
docs/warnings.md. The names are read from the source, not from a list, so a
new flag or warning class without a row fails here:

- the string argument (first or second) of a call to a flag function
  (`numerical_flags.flag`, `flag_numerics`, `ee_follow.flag`, a module's
  own `_flag`), given as a literal or as a name: a top-level string
  constant of the module (FLAG_X = 'name'), one it imports by name from
  another module of the repository (from src.x import FLAG_X), or one read
  off an imported module of the repository (events.FLAG_X);
- the same argument of a call to a helper of the module that hands one of
  its parameters on to a flag function as that argument
  (def _flag_not_finite(values, name, where): flag(name, ...)), at the
  parameter's position or keyword, through helpers of helpers too;
- the value of a 'flag' key in a dict literal ({'flag': name, ...});
- the string key stored into a mapping whose name contains 'flag'
  (flags['name'] = ...);
- every class in src that derives from a ...Warning class.

A name counts as documented when it appears in backticks in the page, alone or
quoted ('name').
"""
import ast
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NAME = re.compile(r'[a-z][a-z0-9]*(_[a-z0-9]+)+$')
FLAG_CALL = re.compile(r'_?flag(_numerics)?$')


def _name_of(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    return ''


def _string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _module_strings(tree):
    """{name: value} of the module's top-level NAME = 'string' assignments."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and _string(node.value) is not None:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    found[t.id] = node.value.value
    return found


def _strings_of(repo, module, cache):
    """The top-level string constants of the repository module `module`
    (dotted), {} when it is not a file of the repository."""
    path = os.path.join(repo, *module.split('.')) + '.py'
    if path not in cache:
        cache[path] = {}
        if os.path.exists(path):
            with open(path) as fh:
                cache[path] = _module_strings(ast.parse(fh.read()))
    return cache[path]


def _named_strings(tree, repo, cache):
    """(names, modules): the string constants a flag call may name, the
    module's own top-level ones and those it imports by name from a module of
    the repository, and {alias: constants} of the repository modules it
    imports whole (from src.x import events; import src.x.events as ev)."""
    found, modules = {}, {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            strings = _strings_of(repo, node.module, cache)
            for alias in node.names:
                if alias.name in strings:
                    found[alias.asname or alias.name] = strings[alias.name]
                elif os.path.exists(os.path.join(repo, *node.module.split('.'), alias.name)
                                    + '.py'):
                    modules[alias.asname or alias.name] = _strings_of(
                        repo, f'{node.module}.{alias.name}', cache)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = _strings_of(repo, alias.name, cache)
    found.update(_module_strings(tree))
    return found, modules


def _flag_arguments(call, forwarders):
    """The arguments of `call` that may hold a flag name: the first two of a
    flag function, the forwarded ones of a helper (`_forwarders`)."""
    name = _name_of(call.func)
    if FLAG_CALL.match(name):
        return list(call.args[:2])
    args = []
    for position, param in forwarders.get(name, ()):
        if position is not None and 0 <= position < len(call.args):
            args.append(call.args[position])
        args.extend(kw.value for kw in call.keywords if kw.arg == param)
    return args


def _forwarders(tree):
    """{function name: [(position, parameter)]} of the module's functions that
    hand a parameter on to a flag function (or to another such function) as
    a flag argument; the position is the parameter's among the call's
    positional arguments (a method's self or cls not counted), None for a
    keyword-only one."""
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    found = {}
    changed = True
    while changed:
        changed = False
        for fn in functions:
            positional = [a.arg for a in fn.args.posonlyargs + fn.args.args]
            offset = 1 if positional[:1] in (['self'], ['cls']) else 0
            keyword_only = [a.arg for a in fn.args.kwonlyargs]
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                for arg in _flag_arguments(call, found):
                    if not isinstance(arg, ast.Name):
                        continue
                    if arg.id in positional:
                        entry = (positional.index(arg.id) - offset, arg.id)
                    elif arg.id in keyword_only:
                        entry = (None, arg.id)
                    else:
                        continue
                    if entry not in found.setdefault(fn.name, []):
                        found[fn.name].append(entry)
                        changed = True
    return found


def _resolved(arg, names, modules):
    """The string a flag argument holds, when the source shows it: a literal,
    a string constant by name, or one read off an imported module."""
    s = _string(arg)
    if s is None and isinstance(arg, ast.Name):
        s = names.get(arg.id)
    if (s is None and isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
            and arg.value.id in modules):
        s = modules[arg.value.id].get(arg.attr)
    return s


def raised_names(repo=REPO):
    """{name: sorted source files} of the flags and warning classes in
    repo/src."""
    found, cache = {}, {}

    def add(name, where):
        found.setdefault(name, set()).add(where)

    for d, _, files in os.walk(os.path.join(repo, 'src')):
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(d, f)
            where = os.path.relpath(path, repo)
            with open(path) as fh:
                text = fh.read()
            if 'flag' not in text and 'Warning' not in text:
                continue                  # the generated modules, mostly
            tree = ast.parse(text)
            names, modules = _named_strings(tree, repo, cache)
            forwarders = _forwarders(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for arg in _flag_arguments(node, forwarders):
                        s = _resolved(arg, names, modules)
                        if s is not None and NAME.match(s):
                            add(s, where)
                elif isinstance(node, ast.Dict):
                    for k, v in zip(node.keys, node.values):
                        if _string(k) == 'flag' and _string(v) is not None:
                            add(_string(v), where)
                elif isinstance(node, (ast.Assign, ast.AugAssign)):
                    targets = (node.targets if isinstance(node, ast.Assign)
                               else [node.target])
                    for t in targets:
                        if (isinstance(t, ast.Subscript)
                                and 'flag' in _name_of(t.value).lower()):
                            s = _string(t.slice)
                            if s is not None and NAME.match(s):
                                add(s, where)
                elif isinstance(node, ast.ClassDef):
                    if any(_name_of(b).endswith('Warning') for b in node.bases):
                        add(node.name, where)
    return {k: sorted(v) for k, v in found.items()}


def documented():
    with open(os.path.join(REPO, 'docs', 'warnings.md')) as f:
        text = f.read()
    return {m.strip("'") for m in re.findall(r'`([^`]+)`', text)}


def test_scanner_sees_each_kind_of_raise():
    names = raised_names()
    for name in ('z_outside', 'orbital_response_not_converged',
                 'davidson_not_converged', 'CphfNotConverged',
                 'adc_roots_not_converged',
                 # raised only through an imported constant (Parquet.events)
                 'parquet_two_body_not_converged'):
        assert name in names, name
    # raised in this file only through its helper _flag_not_finite(values, name, where)
    helper = os.path.join('src', 'SingleReference', 'Parquet', 'spin_orbital_self_energy.py')
    for name in ('parquet_self_energy_not_finite', 'parquet_static_sigma_not_finite'):
        assert helper in names.get(name, ()), (name, names.get(name))


def test_scanner_follows_helpers_and_modules(tmp_path):
    """A flag name passed through a helper (literal, constant, keyword, a
    helper of a helper, a method) or read off an imported module is seen; a
    helper's parameter that never reaches a flag function is not."""
    pkg = tmp_path / 'src' / 'pkg'
    pkg.mkdir(parents=True)
    (pkg / 'ev.py').write_text("FLAG_ATTR = 'probe_attribute_flag'\n")
    (pkg / 'mod.py').write_text(
        "from src.Base.numerical_flags import flag\n"
        "from src.pkg import ev\n"
        "FLAG_CONST = 'probe_constant_flag'\n"
        "def _flag_not_finite(values, name, where):\n"
        "    flag(name, where, RuntimeWarning)\n"
        "def _outer(label, *, which):\n"
        "    _flag_not_finite(None, which, label)\n"
        "def _not_a_flag(values, name):\n"
        "    return [values, name]\n"
        "class Solver:\n"
        "    def _report(self, name):\n"
        "        flag(name, 'x', RuntimeWarning)\n"
        "    def run(self):\n"
        "        self._report('probe_method_flag')\n"
        "def run():\n"
        "    _flag_not_finite(1, 'probe_literal_flag', 'where')\n"
        "    _flag_not_finite(1, FLAG_CONST, 'where')\n"
        "    _flag_not_finite(1, name='probe_keyword_flag', where='w')\n"
        "    _outer('w', which='probe_nested_flag')\n"
        "    flag(ev.FLAG_ATTR, 'x', RuntimeWarning)\n"
        "    _not_a_flag(1, 'probe_not_a_flag')\n")
    names = raised_names(str(tmp_path))
    for name in ('probe_literal_flag', 'probe_constant_flag', 'probe_keyword_flag',
                 'probe_nested_flag', 'probe_method_flag', 'probe_attribute_flag'):
        assert names.get(name) == [os.path.join('src', 'pkg', 'mod.py')], (name, names)
    assert 'probe_not_a_flag' not in names


def test_every_flag_and_warning_class_has_a_row():
    names, docs = raised_names(), documented()
    missing = {k: v for k, v in names.items() if k not in docs}
    assert not missing, (
        'docs/warnings.md does not name these flags or warning classes '
        f'(name: files raising it): {missing}')
