"""The Hedin cycles in zero dimensions (src/SingleReference/HedinCycles, milestone 1).

The cycle loop of schemes.py runs on the 0-D backend (zero_d.py), where every object is a power of
the start line G^(0) times a series in the coupling y = v (G^(0))^2, every diagram has weight +1 and a
closed loop the weight ell. kappa^(n) = G^(0) (Sigma^(n) - Sigma^(0)), and its coefficient of y^k is
the number of the scheme's diagrams at v^k. An independent oracle, tests/hedin_zero_d_oracle.py, holds
the closed-form maps of the talk's backup slide "the maps in zero dimensions".

The reference values are tests/data/hedin_cycles_reference_values.json, a copy of
notes/implementation_tests/reference_values.json of the theory repository (AutomaticEquationGenerator,
working tree at 366bf6d, where the file and its generator src/make_implementation_references.py are
not committed; sha256 of the copy 08397973dc705579...), written there from src/zero_d_convergence.py
with mpmath at 40 digits.

1. Counts at loop weight 1, v^1 .. v^8, cycles 1 .. 5, both starts: the shared loop for all four
   schemes (cycle-based Gamma/W and cycle-based I/L each against the one 'cycle-based' row) and the
   oracle, and the exact rows from the oracle's Wick series.
2. Counts as polynomials in the loop weight, v^1 .. v^6, cycles 1 .. 3: the shared loop at ell = 0 .. 8,
   interpolated exactly, and the exact rows from cycle 7 of SG-I/L, which is exact through v^7.
3. Finite coupling, cycles 1 .. 6: the 0-D complex boson (ell = 1, two couplings) and the Rossi-Werner
   fermions (ell = -2, two couplings), both starts, in 50-digit decimals. kappa^(n) of the shared loop
   and of the oracle to 1e-12 relative (the reference prints 25 digits), and the relative error of G to
   its 6 printed digits. The file's own columns: line/g = 1 + m z and bare x = z/(1 + m z)^2 with
   m = ell from (H) and 1 + ell from (HF), exact kappa = 1 - (line/g)/(exact G/g), one bare coupling and
   one exact G for both starts, and the exact models themselves, G/g = 1/(1 + a) for the fermions and
   the boson's closed form (1 - 1/Z)/x with Z = sqrt(pi)/s erfcx(1/s), s = sqrt(-2x). Then the shared
   loop against the oracle exactly, in fractions, at the coupling -1/10 (cycles 1 .. 4, ell = 1 and -2).
4. The two double-counting traps, built from the same operations, with their numbers read from the
   reference file: SG's Eq. (7) used literally after the Hartree update holds 241 diagrams at v^4 in
   cycle 4 from (H), where 238 exist (and in cycle 5, exact only through v^3 from cycle 3 on), and the
   vertex taken along the start's L^(1) with W also built from P holds 27 at v^3 from cycle 3 on, where
   26 exist. The oracle's closed forms of the traps agree.
5. Tier 1 of the test plan in 0-D: cycle-based Gamma/W equals cycle-based I/L, SG-Gamma/W without its
   Hartree lag equals SG-I/L, the SG identities against the BSE with the kernel along the map (for
   SG-I/L: J^(n-1) L^(n) = L^(1) with the loop's own L^(1), and Sigma^(n) closed on the BSE's L^(n)
   equal to the loop's Sigma^(n); for SG-Gamma/W: Gamma^(n); cycles 2 .. 5), cycle 1 the same in every
   scheme with the closed forms y/(1 - ell y) from (H) and (1 + ell) y^2 / (1 - (1 + ell) y) from (HF)
   at ell = -2, 1 and 3, at cycle 2 SG-I/L minus SG-Gamma/W equal to G^(1) (W^(1) - W^(2)) Gamma^(2)
   (the i of the closure is absorbed in 0-D), and the shared loop equal to the oracle at ell = -2, 1, 3.
6. The known facts: cycle n is exact through v^n from (H) and through v^(n+1) from (HF) and lacks
   diagrams at the next order, and coefficient by coefficient SG-I/L <= SG-Gamma/W <= cycle-based
   <= exact (cycles 1 .. 5, v^1 .. v^8, ell = 1).
7. The backend's rules and the refusals: a sum of quantities carrying different powers of G^(0), an
   exhausted jet, an unknown scheme or start, a cycle count that is not a positive integer, lag=False
   outside SG-Gamma/W, a backend that lacks an operation, and the 0-D helpers refusing before any work
   (a non-integer cycle count, a non-integer loop weight for counts, a zero order).

Run: python tests/test_hedin_cycles_zero_d.py (0.6 s), or under pytest (about 2 s with its start-up).
"""
import json
import math
import os
import re
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.special import erfcx

from src.SingleReference.HedinCycles import zero_d as Z
from src.SingleReference.HedinCycles.schemes import SCHEMES, gw_level, run_cycles, start_level
from tests import hedin_zero_d_oracle as O

REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                         'hedin_cycles_reference_values.json')
#: the reference file's row for each scheme (the two cycle-based routes share one row)
ROW = {'sg_il': 'SG-I/L', 'sg_gw': 'SG-Gamma/W', 'cb_gw': 'cycle-based', 'cb_il': 'cycle-based'}
MODELS = {'l = 1, complex boson, lam < 0': 1,
          'l = -2, Rossi-Werner spin-1/2 fermions, a = |U|/mu^2': -2}
#: the reference file's keys of the two double-counting traps
TRAP_EQ7 = 'Schindlmayr-Godby Eq. (7) used literally after the Hartree update (treats h as a bare line)'
TRAP_L1 = 'vertex taken with respect to the external potential instead of the total one, with W also built from P'
#: relative bar on kappa^(n) at a finite coupling: the test plan's aim
KAPPA_REL_TOL = Decimal('1e-12')
#: relative bar on relations between the reference's 25-digit columns
COLUMN_REL_TOL = Decimal('1e-20')
#: relative bar on the boson's exact G from scipy's erfcx in double precision
BOSON_REL_TOL = 1e-12
DIGITS = 50


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def reference():
    with open(REFERENCE) as fh:
        return json.load(fh)


def interpolate(xs, ys):
    """Exact integer polynomial coefficients through the points (Lagrange in fractions)."""
    n = len(xs)
    coef = [Fraction(0)] * n
    for i in range(n):
        basis, denom = [Fraction(1)], Fraction(1)
        for j in range(n):
            if j == i:
                continue
            basis = [Fraction(0)] + basis
            for k in range(len(basis) - 1):
                basis[k] -= xs[j] * basis[k + 1]
            denom *= xs[i] - xs[j]
        for k in range(n):
            coef[k] += ys[i] * basis[k] / denom
    while len(coef) > 1 and coef[-1] == 0:
        coef.pop()
    assert all(q.denominator == 1 for q in coef), coef
    return [int(q) for q in coef]


def relative(a, b):
    return abs(a - b) / abs(b)


# --------------------------------------------------------------------------- 1
def check_counts_at_loop_weight_one():
    ref = reference()['counts, l = 1 (v^1..v^8, cycles 1..5)']
    ok = True
    for start in ('H', 'HF'):
        ok &= check(O.exact_series(start, 8)[1:9] == ref[start]['exact'],
                    f'({start}) exact series, oracle (Wick) = reference', ref[start]['exact'][:5])
        for scheme in SCHEMES:
            got = Z.counts(scheme, start, 5, 8)
            want = [ref[start][ROW[scheme]][f'cycle {n}'] for n in range(1, 6)]
            ok &= check(got == want, f'({start}) {scheme}: cycles 1-5, v^1..v^8, shared loop = reference',
                        f'cycle 2: {got[1][:5]}')
        for name in O.SCHEMES:
            got = [row[1:9] for row in O.series(name, 5, 8, start)]
            want = [ref[start][name][f'cycle {n}'] for n in range(1, 6)]
            ok &= check(got == want, f'({start}) {name}: oracle = reference')
    return ok


# --------------------------------------------------------------------------- 2
def check_counts_as_polynomials_in_the_loop_weight():
    ref = reference()['counts as polynomials in l (coefficients of l^0, l^1, ...; v^1..v^6; cycles 1..3)']
    ells = list(range(9))
    ok = True
    for start in ('H', 'HF'):
        exact_rows = [Z.counts('sg_il', start, 7, 6, ell)[6] for ell in ells]
        polys = [interpolate(ells, [rows[k] for rows in exact_rows]) for k in range(6)]
        ok &= check(polys == ref[start]['exact'], f'({start}) exact rows in ell (cycle 7 of SG-I/L)',
                    f'v^3: {polys[2]}')
        for scheme in SCHEMES:
            per_ell = [Z.counts(scheme, start, 3, 6, ell) for ell in ells]
            for n in range(1, 4):
                polys = [interpolate(ells, [rows[n - 1][k] for rows in per_ell]) for k in range(6)]
                want = ref[start][f'{ROW[scheme]}, cycle {n}']
                ok &= check(polys == want, f'({start}) {scheme}, cycle {n}: polynomials in ell = reference',
                            f'v^4: {polys[3]}')
    return ok


# --------------------------------------------------------------------------- 3
def check_finite_coupling():
    ref = reference()['finite coupling, cycles 1..6']
    ok = True
    with localcontext() as ctx:
        ctx.prec = DIGITS
        for model, ell in MODELS.items():
            for point, pv in ref[model].items():
                ok &= check_reference_columns(model, ell, point, pv)
                for start in ('H', 'HF'):
                    block = pv[start]
                    z = Decimal(block['z (u from H, w from HF)'])
                    line, g_exact = Decimal(block['line/g']), Decimal(block['exact G/g'])
                    for scheme in SCHEMES:
                        kappas = Z.kappa_at(scheme, start, z, 6, ell)
                        worst_kappa = worst_error = Decimal(0)
                        for n, kappa in enumerate(kappas, start=1):
                            entry = block[ROW[scheme]][f'cycle {n}']
                            worst_kappa = max(worst_kappa, relative(kappa, Decimal(entry['kappa'])))
                            printed = Decimal(entry['rel. error of G'])
                            error = line / ((1 - kappa) * g_exact) - 1
                            half_unit = Decimal(5) * Decimal(10) ** (printed.adjusted() - 6)
                            worst_error = max(worst_error, abs(error - printed) / half_unit)
                        ok &= check(worst_kappa < KAPPA_REL_TOL and worst_error <= 1,
                                    f'ell = {ell}, {point}, ({start}) {scheme}: cycles 1-6',
                                    f'kappa rel {float(worst_kappa):.1e}, G error within '
                                    f'{float(worst_error):.2f} half-units of its 6th digit')
                    worst = max(relative(k, Decimal(block[name][f'cycle {n}']['kappa']))
                                for name in O.SCHEMES
                                for n, k in enumerate(O.point_values(name, z, 6, start, ell), start=1))
                    ok &= check(worst < KAPPA_REL_TOL, f'ell = {ell}, {point}, ({start}) the oracle: '
                                                       f'every scheme, cycles 1-6', f'{float(worst):.1e} relative')
    return ok


def check_reference_columns(model, ell, point, pv):
    """The reference's own columns against each other, and the exact model behind exact G/g."""
    bare = Decimal(pv['bare x'])
    worst = Decimal(0)
    for start in ('H', 'HF'):
        block = pv[start]
        m = ell if start == 'H' else 1 + ell
        z, line = Decimal(block['z (u from H, w from HF)']), Decimal(block['line/g'])
        g_exact, k_exact = Decimal(block['exact G/g']), Decimal(block['exact kappa'])
        worst = max(worst, relative(1 + m * z, line), relative(z / (1 + m * z) ** 2, bare),
                    relative(1 - line / g_exact, k_exact))
    same_g = pv['H']['exact G/g'] == pv['HF']['exact G/g']
    g_exact = Decimal(pv['H']['exact G/g'])
    if ell == -2:            # Rossi-Werner: G = G0 / (1 + a), a = bare x
        model_rel = float(relative(1 / (1 + bare), g_exact))
        model_ok = model_rel < float(COLUMN_REL_TOL)
    else:                    # the complex boson: G/g = (1 - 1/Z)/x, Z = sqrt(pi)/s erfcx(1/s), s = sqrt(-2x)
        x = float(bare)
        s = math.sqrt(-2 * x)
        Zx = math.sqrt(math.pi) / s * erfcx(1 / s)
        model_rel = abs((1 - 1 / Zx) / x / float(g_exact) - 1)
        model_ok = model_rel < BOSON_REL_TOL
    return check(worst < COLUMN_REL_TOL and same_g and model_ok,
                 f'ell = {ell}, {point}: the columns agree (line/g, bare x, exact kappa, one exact G) '
                 f'and exact G/g is the model\'s', f'columns {float(worst):.1e}, model {model_rel:.1e}')


def check_loop_against_oracle_in_fractions(coupling=Fraction(-1, 10), cycles=4):
    ok = True
    for ell in (1, -2):
        for start in ('H', 'HF'):
            same = all(Z.kappa_at(s, start, coupling, cycles, ell)
                       == O.point_values(ROW[s], coupling, cycles, start, ell) for s in SCHEMES)
            ok &= check(same, f'ell = {ell}, ({start}): the shared loop = the oracle exactly at z = {coupling}, '
                              f'every scheme, cycles 1-{cycles}')
    return ok


# --------------------------------------------------------------------------- 4
def trap_kappas(rule, cycles, order=8):
    """SG-Gamma/W from (H) with a wrong vertex, built from the backend's operations:
    'eq7_literal': Gamma^(n) = 1 + (delta Sigma_xc^(n-1) / delta G^(0)) G^(0) Gamma^(1) G^(0) (no Lambda_H),
    'along_l1':    Gamma^(n) = 1 + (delta Sigma_xc^(n-1) / delta G^(0)) L^(1) (the vertex along L^(1))."""
    alg = Z.ZeroDBackend(1, 0, order, 1)
    S0 = start_level(alg, 'H')
    G0 = alg.line0
    legs = alg.vertex_legs(G0, alg.solve_vertex(G0, alg.kernel(S0, G0)))
    L1 = alg.solve_l(G0, alg.kernel(lambda g: alg.sigma_h(g) + S0(g), G0))
    lines, sigmas = [G0], [alg.sigma_h(G0) + S0(G0)]
    for n in range(1, cycles + 1):
        G = lines[-1]
        if n == 1:
            sigma = alg.sigma_h(G) + gw_level(alg, S0)(G)
        else:
            xc = sigmas[n - 1] - alg.sigma_h(lines[n - 2])
            vertex = 1 + alg.along_map(xc, legs if rule == 'eq7_literal' else L1)
            sigma = alg.sigma_h(G) + alg.sigma_gw(G, alg.screen(alg.polarizability(G, vertex)), vertex)
        sigmas.append(sigma)
        lines.append(alg.dyson(sigma - sigmas[0]))
    return [(G0 * (s - sigmas[0])).series() for s in sigmas[1:]]


def check_double_counting_traps():
    traps = reference()['known double-counting traps (0-D counts)']
    exact = O.exact_series('H', 8)
    match = re.fullmatch(r'cycle (\d+) holds (\d+) diagrams at v\^(\d+) from \(H\), where only (\d+) exist',
                         traps[TRAP_EQ7])
    cycle, held, order, exist = map(int, match.groups())
    literal = trap_kappas('eq7_literal', 6)
    ok = check(exact[order] == exist and literal[cycle - 1][order] == held == literal[cycle][order],
               f"SG's Eq. (7) literally after the Hartree update: {held} at v^{order} in cycles {cycle} and "
               f"{cycle + 1} ({exist} exist)", f'cycle {cycle}: {literal[cycle - 1][1:6]}')
    through = [max(k for k in range(1, 9) if row[1:k + 1] == exact[1:k + 1]) for row in literal]
    ok &= check(through[:5] == [1, 2, 3, 3, 3],
                "  ... and it is exact only through v^3 from cycle 3 on", f'exact through {through[:5]}')
    match = re.fullmatch(r'(\d+) diagrams at v\^(\d+) instead of (\d+)', traps[TRAP_L1])
    held, order, exist = map(int, match.groups())
    along_l1 = trap_kappas('along_l1', 6)
    ok &= check(exact[order] == exist and [row[order] == held for row in along_l1] == [False, False] + [True] * 4,
                f'the vertex along L^(1), W from P: {held} at v^{order} from cycle 3 on ({exist} exist)',
                f'v^{order} per cycle: {[row[order] for row in along_l1]}')
    ok &= check(literal == [r[:9] for r in O.trap_series('eq7_literal', 6, 8, 'H')]
                and along_l1 == [r[:9] for r in O.trap_series('along_l1', 6, 8, 'H')],
                "  both traps equal the oracle's closed forms (cycles 1-6, v^0..v^8)")
    return ok


# --------------------------------------------------------------------------- 5
def check_tier_one_in_zero_d():
    ok = True
    for ell in (1, 3):
        for start in ('H', 'HF'):
            gw = Z.kappa_series('cb_gw', start, 5, 8, ell)
            il = Z.kappa_series('cb_il', start, 5, 8, ell)
            ok &= check(gw == il, f'ell = {ell}, ({start}): cycle-based Gamma/W = cycle-based I/L, cycles 1-5')
    with localcontext() as ctx:
        ctx.prec = DIGITS
        for start in ('H', 'HF'):
            gw = Z.kappa_at('cb_gw', start, Decimal('-0.3'), 6, 1)
            il = Z.kappa_at('cb_il', start, Decimal('-0.3'), 6, 1)
            worst = max(relative(b, a) for a, b in zip(gw, il))
            ok &= check(worst < Decimal('1e-45'), f'({start}) the same at z = -0.3, cycles 1-6',
                        f'{float(worst):.1e} relative')
    for ell in (1, 2):
        for start in ('H', 'HF'):
            unlagged = Z.kappa_series('sg_gw', start, 5, 8, ell, lag=False)
            sgil = Z.kappa_series('sg_il', start, 5, 8, ell)
            lagged = Z.kappa_series('sg_gw', start, 5, 8, ell)
            ok &= check(unlagged == sgil and lagged != sgil,
                        f'ell = {ell}, ({start}): SG-Gamma/W without its lag = SG-I/L, with it not')
    for start in ('H', 'HF'):
        ok &= check_sg_shortcut_against_explicit_bse(start)
        ok &= check_cycle_one(start)
        ok &= check_cycle_two_difference(start)
    for ell in (-2, 1, 3):
        for start in ('H', 'HF'):
            same = all(Z.kappa_series(s, start, 5, 8, ell) == O.series(ROW[s], 5, 8, start, ell)
                       for s in SCHEMES)
            ok &= check(same, f'ell = {ell}, ({start}): every scheme of the shared loop = the oracle, cycles 1-5')
    return ok


def check_sg_shortcut_against_explicit_bse(start, cycles=5, order=8):
    """The BSE with the kernel along the map, Xi^(n) = i (delta Sigma^(n-1) / delta G^(0)) J^(n-1) with
    J^(n-1) = (delta G^(n-1) / delta G^(0))^-1, against the SG shortcuts of the loop. For SG-I/L,
    L^(n) = J^(n-1)^-1 L^(1) with the loop's own L^(1), and Sigma^(n) closed on the BSE's L^(n) equal to
    the loop's Sigma^(n). For SG-Gamma/W, Hedin's BSE with the kernel of Sigma_xc^(n-1) equal to the
    loop's Gamma^(n)."""
    alg = Z.ZeroDBackend(1, 0, order, 1)
    one = Z.Quantity.constant(0, 1, 0, order)
    il = run_cycles(alg, 'sg_il', start, cycles)
    gw = run_cycles(alg, 'sg_gw', start, cycles)
    ok_l = ok_sigma = ok_gamma = True
    for n in range(2, cycles + 1):
        G, d_G = il.lines[n - 1], alg.along_map(il.lines[n - 1], one)
        kernel = alg.along_map(il.sigmas[n - 1], one) / d_G
        L_bse = alg.solve_l(G, kernel)
        ok_l &= alg.along_map(G, il.l1).series() == L_bse.series()
        closed = alg.sigma_h(G) + alg.sigma_x(G) + alg.sigma_il(G, alg.apply_kernel(kernel, L_bse))
        ok_sigma &= closed.series() == il.sigmas[n].series()
        G, d_G = gw.lines[n - 1], alg.along_map(gw.lines[n - 1], one)
        xc = gw.sigmas[n - 1] - alg.sigma_h(gw.lines[n - 2])
        explicit = alg.solve_vertex(G, alg.along_map(xc, one) / d_G)
        ok_gamma &= explicit.series() == gw.vertices[n].series()
    return (check(ok_l, f'({start}) SG-I/L: J^(n-1) L^(n) = L^(1) against the BSE, cycles 2-{cycles}')
            & check(ok_sigma, f'({start}) SG-I/L: Sigma^(n) closed on the BSE\'s L^(n) = the loop\'s Sigma^(n)')
            & check(ok_gamma, f'({start}) SG-Gamma/W: Gamma^(n) against the BSE with the kernel along the map'))


def check_cycle_one(start, order=8):
    ok = True
    for ell in (-2, 1, 3):
        rows = [Z.kappa_series(s, start, 1, order, ell)[0] for s in SCHEMES]
        if start == 'H':        # y / (1 - ell y)
            want = [0] + [ell ** (k - 1) for k in range(1, order + 1)]
        else:                   # (1 + ell) y^2 / (1 - (1 + ell) y)
            want = [0, 0] + [(1 + ell) ** (k - 1) for k in range(2, order + 1)]
        ok &= check(all(r == want for r in rows),
                    f'({start}) ell = {ell}: cycle 1 is the same in every scheme and has the closed form', want[:6])
    return ok


def check_cycle_two_difference(start, order=8):
    """At cycle 2 both SG schemes share Gamma^(2), and SG-I/L uses W^(1) where SG-Gamma/W uses W^(2)."""
    alg = Z.ZeroDBackend(1, 0, order, 1)
    S0 = start_level(alg, start)
    G0 = alg.line0
    vertex1 = alg.solve_vertex(G0, alg.kernel(S0, G0))
    W1 = alg.screen(alg.polarizability(G0, vertex1))
    il = run_cycles(alg, 'sg_il', start, 2)
    gw = run_cycles(alg, 'sg_gw', start, 2)
    difference = il.sigmas[2] - gw.sigmas[2]
    formula = gw.lines[1] * (W1 - gw.screened[2]) * gw.vertices[2]
    first = next(k for k, c in enumerate(difference.series()) if c)
    expected = 3 if start == 'H' else 4      # one order past cycle 2's exactness
    return check(difference.series() == formula.series() and first == expected,
                 f'({start}) cycle 2: SG-I/L - SG-Gamma/W = G^(1) (W^(1) - W^(2)) Gamma^(2), '
                 f'from v^{expected} on', f'G^(0) times it begins {(G0 * difference).series()[:6]}')


# --------------------------------------------------------------------------- 6
def check_known_facts(cycles=5, order=8):
    ok = True
    for start in ('H', 'HF'):
        exact = O.exact_series(start, order)
        rows = {s: Z.kappa_series(s, start, cycles, order) for s in ('sg_il', 'sg_gw', 'cb_gw')}
        orders = True
        for s, kappas in rows.items():
            for n, row in enumerate(kappas, start=1):
                last = n if start == 'H' else n + 1
                orders &= row[:last + 1] == exact[:last + 1] and row[last + 1] < exact[last + 1]
        ok &= check(orders, f'({start}) cycle n is exact through v^{"n" if start == "H" else "(n+1)"} '
                            f'and lacks diagrams at the next order, cycles 1-{cycles}')
        ordered = all(a <= b <= c <= e
                      for n in range(cycles)
                      for a, b, c, e in zip(rows['sg_il'][n], rows['sg_gw'][n], rows['cb_gw'][n], exact))
        ok &= check(ordered, f'({start}) SG-I/L <= SG-Gamma/W <= cycle-based <= exact, coefficient by coefficient')
    return ok


# --------------------------------------------------------------------------- 7
def raises(fn, kind=Exception):
    try:
        fn()
    except kind:
        return True
    return False


def check_rules_and_refusals():
    alg = Z.ZeroDBackend(1, 0, 4, 1)
    ok = check(raises(lambda: alg.line0 + alg.v, ValueError),
               'adding quantities that carry different powers of G^(0) is refused')
    ok &= check(raises(lambda: alg.line0 + 1, ValueError), 'adding a number to a line is refused')
    point = Decimal('-0.1')
    ok &= check(raises(lambda: run_cycles(Z.ZeroDBackend(1, point, 0, 2), 'cb_gw', 'HF', 3), ValueError)
                and not raises(lambda: run_cycles(Z.ZeroDBackend(1, point, 0, 3), 'cb_gw', 'HF', 3)),
                'a fixed-form jet shorter than the cycles is refused (from (HF), order 2 for 3 cycles)')
    ok &= check(raises(lambda: run_cycles(Z.ZeroDBackend(1, point, 2, 1), 'sg_il', 'H', 5), ValueError),
                'a coupling jet shorter than the cycles is refused')
    ok &= check(raises(lambda: run_cycles(alg, 'gw', 'H', 2), ValueError), 'an unknown scheme is refused')
    ok &= check(raises(lambda: run_cycles(alg, 'sg_il', 'HF0', 2), ValueError), 'an unknown start is refused')
    ok &= check(raises(lambda: run_cycles(alg, 'sg_il', 'H', 0), ValueError)
                and raises(lambda: run_cycles(alg, 'sg_il', 'H', True), ValueError),
                'zero cycles and a boolean cycle count are refused')
    ok &= check(raises(lambda: run_cycles(alg, 'sg_il', 'H', 2, lag=False), ValueError),
                'lag=False outside SG-Gamma/W is refused')

    class Incomplete:
        line0 = alg.line0
    ok &= check(raises(lambda: run_cycles(Incomplete(), 'sg_il', 'H', 2), TypeError),
                'a backend without every operation is refused')
    ok &= check(raises(lambda: Z.kappa_at('sg_il', 'H', point, 2.5), ValueError),
                'kappa_at refuses a non-integer cycle count before building a backend')
    ok &= check(raises(lambda: Z.counts('sg_il', 'H', 2, 4, Fraction(1, 2)), ValueError)
                and raises(lambda: Z.counts('sg_il', 'H', 2, 0), ValueError),
                'counts refuses a non-integer loop weight and a zero order before any cycle')
    return ok


def run():
    print('-- 1. counts at loop weight 1')
    ok = check_counts_at_loop_weight_one()
    print('\n-- 2. counts as polynomials in the loop weight')
    ok &= check_counts_as_polynomials_in_the_loop_weight()
    print('\n-- 3. finite coupling')
    ok &= check_finite_coupling()
    ok &= check_loop_against_oracle_in_fractions()
    print('\n-- 4. the two double-counting traps')
    ok &= check_double_counting_traps()
    print('\n-- 5. Tier 1 in zero dimensions')
    ok &= check_tier_one_in_zero_d()
    print('\n-- 6. exactness orders and the order of the counts')
    ok &= check_known_facts()
    print('\n-- 7. rules and refusals')
    ok &= check_rules_and_refusals()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_zero_d():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
