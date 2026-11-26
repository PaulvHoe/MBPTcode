"""The four vertex schemes, the two starts and the cycle loop of the Hedin cycles.

Written only against the operations of algebra.py, so the same loop runs on every backend. The
equations, with Sigma^(0) the start's self-energy and G^(n) = G^(0) + G^(0) [Sigma^(n) - Sigma^(0)] G^(n):

Gamma/W route:  Sigma^(n) = Sigma_H[G^(n-1)] + i G^(n-1) W^(n) Gamma^(n),
                P^(n) = -i G^(n-1) G^(n-1) Gamma^(n),  W^(n) = v + v P^(n) W^(n),
                Gamma^(n) = 1 + (delta Sigma_xc^(n-1) / delta G^(n-1)) G^(n-1) G^(n-1) Gamma^(n).
I/L route:      Sigma^(n) = Sigma_HF[G^(n-1)] + i v G^(n-1) Xi^(n) L^(n),
                L^(n) = L_0^(n) + L_0^(n) Xi^(n) L^(n),  L_0^(n) = -i G^(n-1) G^(n-1),
                Xi^(n) = i delta Sigma^(n-1) / delta G^(n-1).

The derivative delta Sigma^(n-1) / delta G^(n-1) is read in one of two ways.

At fixed form (the cycle-based schemes) it is the derivative of the previous cycle's formula with
respect to its own line. The formulas are Hedin's level functionals of a generic line G,
S_k[G] = i G W_k[G] Gamma_k[G] with Gamma_k = 1 + (delta S_(k-1) / delta G) G G Gamma_k (Gamma/W
route), or S_k[G] = Sigma_x[G] + i v G Xi_k L_k with Xi_k = i delta(Sigma_H + S_(k-1)) / delta G
(I/L route), and cycle n evaluates Sigma^(n) = Sigma_H[G^(n-1)] + S_n[G^(n-1)]. Since S_(k-1)
rebuilds its screening from its own argument, the kernel of cycle n holds W[G^(n-1)].

Along the map (the SG schemes) it is the derivative through the start line, which every
Sigma^(n-1) depends on explicitly, composed with J^(n-1) = delta G^(0) / delta G^(n-1). Schindlmayr and
Godby's construction then solves the BSE explicitly. Their Eq. (7) is its Gamma/W form with the Hartree
term frozen at that of G^(0). The group note extends it to the Hartree update, through the factor
[1 + Lambda_H]^-1 in J^(n-1) G^(n-1) Gamma^(n) G^(n-1) = G^(0) Gamma^(1) G^(0) [1 + Lambda_H^(n-1)]^-1,
and to the I/L route, through J^(n-1) L^(n) = L^(1):
    SG-I/L:      Sigma^(n) = Sigma_HF[G^(n-1)] + i v G^(n-1) Xi~^(n) L^(1),
                 Xi~^(n) L^(1) = i (delta Sigma^(n-1) / delta G^(0)) L^(1);
    SG-Gamma/W:  Gamma^(n) = 1 + Lambda^(n-1) [1 + Lambda_H^(n-1)]^-1,
                 Lambda^(n-1) = (delta Sigma_xc^(n-1) / delta G^(0)) G^(0) Gamma^(1) G^(0),
                 Lambda_H^(n-1) = (delta (Sigma_H[G^(n-2)] - Sigma_H[G^(0)]) / delta G^(0)) G^(0) Gamma^(1) G^(0),
with Sigma_xc^(n-1) = Sigma^(n-1) - Sigma_H[G^(n-2)]. The Hartree term inside Sigma^(n-1) was built
from G^(n-2), and Lambda_H carries this one-cycle lag. Only the BSE of the start (L^(1), or the
vertex Gamma^(1)) is solved.

Starts: 'H', the self-consistent Hartree line h with Sigma^(0) = Sigma_H[h], L^(1) the RPA and
Gamma^(1) = 1, and 'HF', the self-consistent Hartree-Fock line f with Sigma^(0) = Sigma_HF[f], L^(1)
the TDHF response and Gamma^(1) the exchange ladder. Cycle 1 is the same in all four schemes: G0W0 on
h, or GW with the TDHF vertex and screening on f.
"""
from dataclasses import dataclass, field

from src.Base.numerical_flags import collecting, gathered

from src.SingleReference.HedinCycles.algebra import missing_operations

SCHEMES = ('sg_il', 'sg_gw', 'cb_gw', 'cb_il')
SCHEME_NAMES = {'sg_il': 'SG-I/L', 'sg_gw': 'SG-Gamma/W',
                'cb_gw': 'cycle-based Gamma/W', 'cb_il': 'cycle-based I/L'}
STARTS = ('H', 'HF')
ROUTE = {'sg_il': 'il', 'cb_il': 'il', 'sg_gw': 'gw', 'cb_gw': 'gw'}


def start_level(alg, start):
    """S_0[G], the exchange-correlation formula of the start: zero from 'H', Sigma_x[G] from 'HF'."""
    if start == 'H':
        return lambda G: alg.zero_self_energy()
    return alg.sigma_x


def gw_level(alg, S_prev):
    """S_k[G] = i G W_k Gamma_k, Gamma_k = 1 + (delta S_(k-1) / delta G) G G Gamma_k,
    P_k = -i G G Gamma_k, W_k = v + v P_k W_k: the level functional of the Gamma/W route."""
    def S(G):
        vertex = alg.solve_vertex(G, alg.kernel(S_prev, G))
        W = alg.screen(alg.polarizability(G, vertex))
        return alg.sigma_gw(G, W, vertex)
    return S


def il_level(alg, S_prev):
    """S_k[G] = Sigma_x[G] + i v G Xi_k L_k, Xi_k = i delta(Sigma_H + S_(k-1)) / delta G,
    L_k = L_0 + L_0 Xi_k L_k: the level functional of the I/L route."""
    def S(G):
        K = alg.kernel(lambda g: alg.sigma_h(g) + S_prev(g), G)
        return alg.sigma_x(G) + alg.sigma_il(G, alg.apply_kernel(K, alg.solve_l(G, K)))
    return S


@dataclass
class CycleHistory:
    """The cycles of one run: lines[n] = G^(n) and sigmas[n] = Sigma^(n) for n = 0 .. cycles.

    For SG-Gamma/W, vertices[n] and screened[n] hold Gamma^(n) and W^(n) from cycle 2 on (None
    before), and legs holds the start's G^(0) Gamma^(1) G^(0). For SG-I/L, l1 holds the start's L^(1).
    For every other scheme these stay None. numerical_flags holds the named flags raised during the run
    (`src.Base.numerical_flags`), each distinct one once.
    """
    scheme: str
    start: str
    lines: list = field(default_factory=list)
    sigmas: list = field(default_factory=list)
    vertices: list = field(default_factory=list)
    screened: list = field(default_factory=list)
    legs: object = None
    l1: object = None
    numerical_flags: list = field(default_factory=list)


def validate(scheme, start, cycles, lag=True):
    """Refuse a request run_cycles cannot answer, before any work is done."""
    if scheme not in SCHEMES:
        raise ValueError(f'unknown scheme {scheme!r}: choose one of {SCHEMES}')
    if start not in STARTS:
        raise ValueError(f'unknown start {start!r}: choose one of {STARTS}')
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError(f'cycles must be a positive integer, got {cycles!r}')
    if not lag and scheme != 'sg_gw':
        raise ValueError('lag=False is a variant of SG-Gamma/W only')


def run_cycles(alg, scheme, start, cycles, lag=True):
    """Cycles 1 .. cycles of one scheme (one of SCHEMES) from one start (one of STARTS) on the
    backend alg, returned as a CycleHistory.

    lag=False takes SG-Gamma/W's vertex with respect to the Hartree term of G^(n-1) itself
    (Sigma_H[G^(n-1)] in place of Sigma_H[G^(n-2)] in both Lambda pieces). That cycle is SG-I/L
    exactly (the test of the lag), and lag=False is refused for the other schemes. A backend built on
    a start line (one with start_name, the Matsubara backend) refuses any other start.
    """
    validate(scheme, start, cycles, lag)
    missing = missing_operations(alg)
    if missing:
        raise TypeError(f'the backend lacks the operations {missing}')
    built_on = getattr(alg, 'start_name', None)
    if built_on is not None and built_on != start:
        raise ValueError(f'the backend was built on the {built_on!r} start line, so its cycles need '
                         f'start={built_on!r}, not {start!r} (its line already holds that start\'s '
                         f'static self-energy)')

    flags = []
    with collecting(flags):
        hist = _run(alg, scheme, start, cycles, lag)
    hist.numerical_flags = gathered(flags)
    return hist


def _run(alg, scheme, start, cycles, lag):
    S0 = start_level(alg, start)
    G0 = alg.line0
    hist = CycleHistory(scheme, start, lines=[G0], sigmas=[alg.sigma_h(G0) + S0(G0)],
                        vertices=[None], screened=[None])
    route = ROUTE[scheme]
    level_of = gw_level if route == 'gw' else il_level
    levels = [S0]
    if scheme == 'sg_gw':      # the start's BSE: the vertex Gamma^(1), with a start line on each leg
        hist.legs = alg.vertex_legs(G0, alg.solve_vertex(G0, alg.kernel(S0, G0)))
    if scheme == 'sg_il':      # the start's BSE: L^(1)
        hist.l1 = alg.solve_l(G0, alg.kernel(lambda g: alg.sigma_h(g) + S0(g), G0))

    for n in range(1, cycles + 1):
        G = hist.lines[n - 1]
        vertex = W = None
        if n == 1 or scheme in ('cb_gw', 'cb_il'):
            # cycle 1 (every scheme) and the cycle-based schemes: the level functional at G^(n-1)
            levels.append(level_of(alg, levels[-1]))
            sigma = alg.sigma_h(G) + levels[-1](G)
        elif scheme == 'sg_il':
            sigma = (alg.sigma_h(G) + alg.sigma_x(G)
                     + alg.sigma_il(G, alg.along_map(hist.sigmas[n - 1], hist.l1)))
        else:                  # sg_gw
            # the Hartree term inside Sigma^(n-1), built from G^(n-2) (lag=False: that of G^(n-1))
            hartree = alg.sigma_h(hist.lines[n - 2] if lag else G)
            lam = alg.along_map(hist.sigmas[n - 1] - hartree, hist.legs)
            lam_h = alg.along_map(hartree - alg.sigma_h(G0), hist.legs)
            vertex = alg.sg_vertex(lam, lam_h)
            W = alg.screen(alg.polarizability(G, vertex))
            sigma = alg.sigma_h(G) + alg.sigma_gw(G, W, vertex)
        hist.sigmas.append(sigma)
        hist.lines.append(alg.dyson(sigma - hist.sigmas[0]))
        hist.vertices.append(vertex)
        hist.screened.append(W)
    return hist
