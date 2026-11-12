"""The static Hermitian self-energies of qsPA, closed shell (Quack's
R_Parquet_self_energy_SRG_sp and R_Parquet_self_energy_eta).

qsPA (Marie and Loos, J. Chem. Phys. 163, 194115 (2025), Sec. VII C) replaces
the frequency-dependent parquet self-energy of Eqs. (48), (50) and (51) by a
static matrix Sigma_pq, symmetrized and added to the Fock matrix, whose
eigenvectors are the next one-body iteration's orbitals. Both routines here
return the full matrix before that symmetrization (Quack's R_qsParquet takes
0.5 (Sigma + Sigma^T) itself) and the renormalization factors Z = 1.

The matrix is the sum of the second-order part, the eh part (singlet and
triplet channels, M^eh and Omega^eh) and the pp part (singlet and triplet
channels, the 2p branch M^ee, Omega^ee and the 2h branch M^hh, Omega^hh), with
the closed-shell spin weights Quack hard-codes. Every term of Eqs. (50)-(51)
has a numerator <pr|st> M M* and two denominators: a bosonic one d
(eps_a - eps_i -+ Omega_n, Omega_m - eps_i - eps_j, ...) and a frequency one
omega - E (E = eps_j - Omega_n, eps_i + eps_j - eps_a, ...). The two routines
make the sum static in two ways:

* SRG (reg_PA = .true., the qsPA of the paper's Table III, which the paper
  says is inspired by the SRG-qsGW self-energy of Marie and Loos, J. Chem.
  Theory Comput. 19, 3943 (2023), and gives in its supplementary material;
  what is ported is Quack's routine). Term by term, with flow parameter s
  (`eta`),

      1/d                        ->  R(d) = (1 - exp(-2 s d^2)) / d,
      1/(omega - E) at e_p, e_q  ->  K(D_p, D_q)
                                   = (D_p + D_q) / (D_p^2 + D_q^2)
                                     * (1 - exp(-s (D_p^2 + D_q^2))),

  D_p = eps_p - E, D_q = eps_q - E: the SRG-qsGW kernel, symmetric in p and
  q, regular at every s, with K(D, D) = R(D). The second-order part takes K
  alone. s -> infinity is the unregularized limit, s = 0 gives Sigma = 0.
  At p = q this is the evPA regularization of Eq. (53) on every term: the
  diagonal is Quack's R_Parquet_self_energy_diag_omega at omega = eps_p and
  the same s.

* eta (reg_PA = .false.). The two terms of a group that share a bosonic
  denominator d are recombined, which cancels it,
  [(omega - A)^-1 - (omega - B)^-1] / d = [(omega - A)(omega - B)]^-1 for
  A - B = d, so those terms carry two frequency denominators; then every
  denominator x -> x / (x^2 + eta^2), the real part of its i eta broadened
  form, the frequency ones at omega = eps_p (the row index). Not symmetric
  in p and q; eta = 0 is the unregularized limit.

In the unregularized limits the two diagonals are the same, Quack's dynamic
Sigma_pp(omega = eps_p).

Which SRG form. R_qsParquet calls R_Parquet_self_energy_SRG_sp, the form
ported here (its R_Parquet_self_energy_SRG_dp call is commented out,
R_qsParquet.f90:838-839), while the spin-orbital G_qsParquet calls
G_Parquet_self_energy_SRG_dp (`spin_orbital_self_energy`). Their symmetrized
matrices agree on the diagonal as s grows but not off it, at any s: on the
same converged water/6-31G spectra the off-diagonal elements differ by
1.0e-2 Ha at s = 50 and 9.7e-2 Ha at s = 1e8, and the whole qsPA HOMO by
21 meV at the paper's settings. So the paper's closed-shell qsPA is this
module's, through `restricted.run_parquet`; the spin-orbital qsPA of a
closed shell is another approximation.

Cost. The SRG kernel K couples p and q, so the frequency denominator is not a
product of a p-part and a q-part: every term is a sum over configurations x,

    Sigma_pq += sum_x A_px B_qx K(eps_p + D_x, eps_q + D_x),

whose A and B are built by BLAS from intermediates (the int3 and int4 arrays
of Quack's routine, O(K^6) operations and O(K^4) memory, as the paper says)
and whose kernel is evaluated element by element, exactly: O(K^2 n_x), the
largest n_x being o v^2 (o, v occupied and virtual counts). Configurations
that share a kernel are summed into one A, B pair. The eta form is separable
(its denominators depend on p but not on q), so it is BLAS throughout. At
CH4/aug-cc-pVTZ size (norb = 138, o = 5) the SRG form is 5.2e11
multiply-adds and 7.0e9 kernel elements, the eta form 3.0e11 multiply-adds,
and each allocates at most 0.7 GB (about seven o v^2 norb arrays) beyond its
inputs (eri 2.9 GB, the 2p amplitudes 2.7 GB); on one core of an Apple M4
shared with other jobs they took 167 s (53 s of it the kernel) and 39 s.

Conventions. eri[p, q, r, s] = <pq|rs> (MO, physicist, not antisymmetrized),
occupied orbitals first, no frozen core or virtuals. The spectra are a
`spectra.RestrictedSpectra`; rho[p, q, n] is Quack's rho(p, q, n) (the eh
`eh_sing_rho`, the pp `ee_sing_rho` and `hh_sing_rho`, ...), om its
eigenvalues. Every array is indexed as Quack's (0-based), and the result
matches Quack's SigC element by element, but where Quack's SRG formula has an
exact 0/0: a denominator exactly zero gives the limit, R(0) = 0 and
K(0, 0) = 0 (at D_p^2 + D_q^2 = 0), where Quack's is NaN, so a zero mode
weighs nothing whether rounding puts it at 0 or a few ulps off. The eta
form's x / (x^2 + eta^2) is 0 at x = 0 for every eta whose square does not
underflow (eta above about 1.6e-162 Ha), and 0/0 there, as in Quack, at
eta = 0, its unregularized limit (which the drivers refuse), and at an eta
below that bound (which they take: their broadening is any finite eta > 0).
An element that is not finite (an input that is not finite, an overflow, or
that pole of the eta form) raises the named flag
'parquet_static_sigma_not_finite' (`events.ParquetNotFinite`).
"""
import numpy as np

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import PARQUET_SRG_PAIR_BLOCK_ELEMS, PARQUET_STATIC_PP_BLOCK_ELEMS
from src.Base.utils.parallel.threads import (blas_single_threaded, resolve_workers,
                                             row_map)
from src.SingleReference.Parquet.events import FLAG_STATIC_SIGMA_NOT_FINITE, ParquetNotFinite
from src.SingleReference.Parquet.pairs import EH_SPIN_WEIGHTS, PP_SPIN_WEIGHTS, pp_pair_count
from src.SingleReference.Parquet.spectra import RestrictedSpectra


def static_self_energy_srg(eta, eps, eri, nO, spectra, *, n_workers=None):
    """(sigc, z): the SRG-regularized static parquet self-energy of qsPA.

    Ports R_Parquet_self_energy_SRG_sp (the routine R_qsParquet calls when
    reg_PA = .true.): the second-order part, Eq. (48), and the eh and pp
    parts, Eqs. (50)-(51), term by term, each term's bosonic denominator d
    regularized as R(d) = (1 - exp(-2 eta d^2)) / d and its frequency
    denominator, at the pair (e_p, e_q), as the SRG kernel

        K(D_p, D_q) = (D_p + D_q) / (D_p^2 + D_q^2) (1 - exp(-eta (D_p^2 + D_q^2)))

    (the module docstring has both). This is the form R_qsParquet uses; the
    spin-orbital G_qsParquet uses G_Parquet_self_energy_SRG_dp, whose
    symmetrized matrix differs from this one off the diagonal even as eta
    grows (module docstring, 'Which SRG form'), so the paper's closed-shell
    qsPA comes from this routine. Every element is Quack's; the terms are
    grouped by their kernel's configurations:

    * (i, j, a), D = eps_a - eps_i - eps_j: the 2h1p second-order term, eh
      part 2 and pp part 2 of both spins;
    * (i, a, b), D = eps_i - eps_a - eps_b: the 2p1h second-order term, eh
      part 4 and pp part 4;
    * (j, n), D = Omega^eh_n - eps_j and (b, n), D = -eps_b - Omega^eh_n: eh
      parts 1 and 3 of each spin;
    * (k, n), D = eps_k - Omega^ee_n and (c, n), D = eps_c - Omega^hh_n: pp
      parts 1 and 3 of each spin.

    Cost at norb = K, o occupied, v virtual: the intermediates are GEMMs of
    O(K o v^2 n_vv) (pp parts 1 and 4, the largest) and O(K o v^2 n_S); the
    kernel is K^2 (o v^2 + 2 v n_S + o n_vv,s + o n_vv,t + ...) exact element
    evaluations, about 7 passes each, cut over p in `n_workers` threads.
    Memory: a few K o v^2 arrays and the inputs; eri is read whole, without a
    copy, when it is C-contiguous.

    Parameters
    ----------
    eta : float
        The one-body flow parameter s (Quack's eta_1b), Hartree^-2, >= 0.
        Large is weakly regularized; 0 gives Sigma = 0.
    eps : ndarray, shape (norb,)
        One-body energies eQP (the quasiparticle energies of the current
        one-body iteration), occupied first.
    eri : ndarray, shape (norb, norb, norb, norb)
        <pq|rs> in the orbitals of eps (Quack's ERI_MO).
    nO : int
        Doubly occupied orbitals.
    spectra : RestrictedSpectra
        The converged two-body spectra: eh singlet and triplet (n_S = o v
        each), ee singlet and triplet (v(v+1)/2, v(v-1)/2), hh singlet and
        triplet (o(o+1)/2, o(o-1)/2), each Channel(om, rho[p, q, n]).
    n_workers : int, optional
        Threads for the kernel; None takes OMP_NUM_THREADS, else the CPUs
        of this process (`threads.resolve_workers`). The result is the
        same bits at every count (checked at 1, 2, 3 and 8 on OpenBLAS).

    Returns
    -------
    sigc : ndarray, shape (norb, norb)
        Quack's SigC(p, q), Hartree, not symmetrized.
    z : ndarray, shape (norb,)
        Renormalization factors: 1 (Quack's Z(:) = 1).
    """
    eta, eps, eri, nO, chan = _checked(eta, eps, eri, nO, spectra)
    norb = len(eps)
    o, v = slice(0, nO), slice(nO, norb)
    eo, ev = eps[o], eps[v]
    workers = resolve_workers(n_workers, norb)
    sig = np.zeros((norb, norb))

    # The right-hand factors B of the two orbital families, split by which
    # integral multiplies them on the left: x by <pa|ij> (i, j, a) and
    # <pi|ab> (i, a, b), y by the exchanged <pa|ji> and <pi|ba>. Indexed
    # [q, i, j, a] and [q, i, a, b].
    bx_ija = np.zeros((norb, nO, nO, len(ev)))
    by_ija = 2.0 * eri[o, o, :, v].transpose(2, 1, 0, 3) - eri[o, o, v, :].transpose(3, 1, 0, 2)
    bx_iab = np.zeros((norb, nO, len(ev), len(ev)))
    by_iab = 2.0 * eri[v, v, :, o].transpose(2, 3, 1, 0) - eri[v, v, o, :].transpose(3, 2, 1, 0)

    for spin in ('singlet', 'triplet'):
        _eh_srg(sig, eps, eri, nO, chan['eh_' + spin], EH_SPIN_WEIGHTS[spin], eta, workers,
                bx_ija, by_ija, bx_iab, by_iab)
    for spin in ('singlet', 'triplet'):
        _pp_srg(sig, eps, eri, nO, chan['ee_' + spin], chan['hh_' + spin], PP_SPIN_WEIGHTS[spin],
                eta, workers, bx_ija, bx_iab)

    # (i, j, a): sum_ija [<pa|ij> Bx_qija + <pa|ji> By_qija] K; the kernel
    # is symmetric in i <-> j, so the exchanged term is <pa|ij> By_qjia.
    left = eri[:, v, o, o].transpose(0, 2, 3, 1).reshape(norb, -1)
    right = (bx_ija + by_ija.transpose(0, 2, 1, 3)).reshape(norb, -1)
    del bx_ija, by_ija
    shift = (ev[None, None, :] - eo[:, None, None] - eo[None, :, None]).ravel()
    _srg_pair_sum(sig, eps, shift, left, right, eta, workers)
    # (i, a, b): likewise with a <-> b.
    left = eri[:, o, v, v].reshape(norb, -1)
    right = (bx_iab + by_iab.transpose(0, 1, 3, 2)).reshape(norb, -1)
    del bx_iab, by_iab
    shift = (eo[:, None, None] - ev[None, :, None] - ev[None, None, :]).ravel()
    _srg_pair_sum(sig, eps, shift, left, right, eta, workers)

    _flag_not_finite(sig, 'static_self_energy_srg')
    return sig, np.ones(norb)


def static_self_energy_eta(eta, eps, eri, nO, spectra):
    """(sigc, z): the eta-broadened static parquet self-energy, Sigma_pq(e_p).

    Ports R_Parquet_self_energy_eta (the routine R_qsParquet calls when
    reg_PA = .false.): the second-order part, Eq. (48), and the eh and pp
    parts, Eqs. (50)-(51), at omega = eps_p, the two terms of each group that
    share a bosonic denominator recombined into one with two frequency
    denominators (the bosonic one cancels), every denominator x replaced by
    x / (x^2 + eta^2), the real part of its i eta broadened form. Unlike the
    SRG form the frequency denominators depend on p alone, so each term is

        Sigma_pq += sum_x [A_px L(eps_p + D_x) ...] B_qx,

    one GEMM after its intermediates: O(K o v^2 n_vv) operations (the 3p2h
    term of the 2p branch, the largest) and a few K o v^2 arrays.

    The eh part has Quack's six terms per spin (three 3h2p, three 3p2h), the
    pp part six per spin (4h1p, 3h2p with the 2p and the 2h branch, 4p1h,
    3p2h with each branch), with the weights of pairs.EH_SPIN_WEIGHTS and
    pairs.PP_SPIN_WEIGHTS.

    Parameters
    ----------
    eta : float
        The broadening, Hartree, >= 0; 0 is unregularized. Quack passes its
        eta_1b (reg_1b) here, the variable that is the SRG flow parameter
        under reg_PA; the drivers take it as their own `broadening_1b`.
    eps, eri, nO, spectra
        As static_self_energy_srg.

    Returns
    -------
    sigc : ndarray, shape (norb, norb)
        Quack's SigC(p, q), Hartree, not symmetrized.
    z : ndarray, shape (norb,)
        Renormalization factors: 1 (Quack's Z(:) = 1).
    """
    eta, eps, eri, nO, chan = _checked(eta, eps, eri, nO, spectra)
    norb = len(eps)
    o, v = slice(0, nO), slice(nO, norb)
    eo, ev = eps[o], eps[v]
    sig = np.zeros((norb, norb))

    def lor(d):
        return d / (d * d + eta * eta)

    # The frequency denominators that hold two orbital energies besides e_p:
    # l_ija[p, i, j, a] = L(e_p + e_a - e_i - e_j), l_iab[p, i, a, b] =
    # L(e_p + e_i - e_a - e_b), shared by every part.
    l_ija = lor(eps[:, None, None, None] + (ev[None, None, :] - eo[:, None, None]
                                            - eo[None, :, None])[None])
    l_iab = lor(eps[:, None, None, None] + (eo[:, None, None] - ev[None, :, None]
                                            - ev[None, None, :])[None])

    # second order, 2h1p: sum_ija <pa|ji> L(e_p + e_a - e_i - e_j) (2<ji|qa> - <ji|aq>)
    left = eri[:, v, o, o].transpose(0, 3, 2, 1) * l_ija
    right = 2.0 * eri[o, o, :, v].transpose(2, 1, 0, 3) - eri[o, o, v, :].transpose(3, 1, 0, 2)
    sig += left.reshape(norb, -1) @ right.reshape(norb, -1).T
    # 2p1h: sum_iab <pi|ba> L(e_p + e_i - e_a - e_b) (2<ba|qi> - <ba|iq>)
    left = eri[:, o, v, v].transpose(0, 1, 3, 2) * l_iab
    right = 2.0 * eri[v, v, :, o].transpose(2, 3, 1, 0) - eri[v, v, o, :].transpose(3, 2, 1, 0)
    sig += left.reshape(norb, -1) @ right.reshape(norb, -1).T
    del left, right

    for spin in ('singlet', 'triplet'):
        _eh_eta(sig, eps, eri, nO, chan['eh_' + spin], EH_SPIN_WEIGHTS[spin], lor, l_ija, l_iab)
    for spin in ('singlet', 'triplet'):
        _pp_eta(sig, eps, eri, nO, chan['ee_' + spin], chan['hh_' + spin], PP_SPIN_WEIGHTS[spin],
                lor, l_ija, l_iab)
    _flag_not_finite(sig, 'static_self_energy_eta')
    return sig, np.ones(norb)


# --------------------------------------------------------------------------
# SRG form
# --------------------------------------------------------------------------

def _srg_inverse(d, eta):
    """R(d) = (1 - exp(-2 eta d^2)) / d, Quack's reg1/dem1; at d = 0 exactly
    its limit 0, where Quack's is 0/0 (eta is finite here, `_checked`; inf,
    the bare 1/d, would keep its pole)."""
    zero = d == 0.0
    if not (eta < np.inf and zero.any()):
        return (1.0 - np.exp(-2.0 * eta * d * d)) / d
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (1.0 - np.exp(-2.0 * eta * d * d)) / d
    r[zero] = 0.0                           # Quack's 0/0: the limit, 2 eta d -> 0
    return r


def _srg_pair_sum(sig, eps, shift, left, right, eta, workers):
    """sig[p, q] += sum_x left[p, x] right[q, x] K(eps_p + shift_x, eps_q + shift_x).

    K(a, b) = (a + b) / (a^2 + b^2) (1 - exp(-eta (a^2 + b^2))), Quack's
    reg2 (dem2_p + dem2_q) / dem2_pq, evaluated exactly for every (p, q, x)
    with exp(-eta (a^2 + b^2)) = exp(-eta a^2) exp(-eta b^2): seven passes
    over (q, x) per row p; where a^2 + b^2 is exactly zero (Quack's 0/0) K is
    its limit 0. The rows p run over `workers` threads; each
    thread takes the configurations in blocks, forms a block's per-q factors
    once and runs all its rows over it while it is in cache. A row is summed
    over the blocks in one order whichever thread runs it, so the split
    does not change the bits (with more than one worker BLAS is held at one
    thread under them; one worker leaves BLAS's own pool to its GEMVs, whose
    bits OpenBLAS keeps at any count).
    """
    norb, nx = left.shape
    if nx == 0:
        return
    block = max(1, PARQUET_SRG_PAIR_BLOCK_ELEMS // norb)
    width = min(block, nx)

    def rows(p0, p1):
        part = np.zeros((p1 - p0, norb))
        bufs = np.empty((6, norb, width))
        for x0 in range(0, nx, block):
            x1 = min(x0 + block, nx)
            d, d2, damp, damped, s, t = (b[:, :x1 - x0] for b in bufs)
            r = right[:, x0:x1]
            np.add(eps[:, None], shift[None, x0:x1], out=d)        # eps_q + shift_x
            np.multiply(d, d, out=d2)
            zeros = eta < np.inf and not d2.all()   # a^2 + b^2 = 0 can occur in this block
            np.multiply(d2, -eta, out=damp)
            np.exp(damp, out=damp)
            np.multiply(r, damp, out=damped)
            for p in range(p0, p1):
                np.add(d2, d2[p], out=s)                            # a^2 + b^2
                np.add(d, d[p], out=t)                              # a + b
                if zeros:
                    hit = s == 0.0                                  # Quack's 0/0: K's limit 0
                    s[hit], t[hit] = 1.0, 0.0
                np.divide(t, s, out=t)
                np.multiply(damped, damp[p], out=s)
                np.subtract(r, s, out=s)                            # right (1 - e^-eta(a^2+b^2))
                np.multiply(t, s, out=t)
                part[p - p0] += t @ left[p, x0:x1]
        return part

    if workers > 1:
        with blas_single_threaded(min_threads=1):
            parts = row_map(rows, norb, workers)
    else:
        parts = [rows(0, norb)]
    sig += np.concatenate(parts, axis=0)


def _eh_srg(sig, eps, eri, nO, chan, weights, eta, workers, bx_ija, by_ija, bx_iab, by_iab):
    """One eh spin channel of R_Parquet_self_energy_SRG_sp ('singlet/triplet
    eh part 1-4'): parts 1 and 3 into sig, parts 2 and 4 (int4) into the B
    factors of the (i, j, a) and (i, a, b) families."""
    om, rho = chan
    norb = len(eps)
    nV = norb - nO
    o, v = slice(0, nO), slice(nO, norb)
    eo, ev = eps[o], eps[v]
    wd, wx = weights
    ns = len(om)
    dia = ev[None, :] - eo[:, None]
    # g1[(i,a), n] = rho(i,a,n) R(e_a - e_i - Om_n), g2[(i,a), n] = rho(a,i,n) R(e_a - e_i + Om_n)
    g1 = (rho[o, v, :] * _srg_inverse(dia[:, :, None] - om, eta)).reshape(nO * nV, ns)
    g2 = (rho[v, o, :].transpose(1, 0, 2) * _srg_inverse(dia[:, :, None] + om, eta)).reshape(nO * nV, ns)

    # part 1: int3(p,j,n) = sum_ia (wd <pa|ij> + wx <pa|ji>) g1 + (wd <pi|aj> + wx <pi|ja>) g2
    n1 = wd * eri[:, v, o, o].transpose(0, 3, 2, 1)
    n1p = wd * eri[:, o, v, o].transpose(0, 3, 1, 2)
    if wx:
        n1 = n1 + wx * eri[:, v, o, o].transpose(0, 2, 3, 1)
        n1p = n1p + wx * eri[:, o, o, v].transpose(0, 2, 1, 3)
    int3 = n1.reshape(norb * nO, -1) @ g1 + n1p.reshape(norb * nO, -1) @ g2
    del n1, n1p
    # Sigma_pq += sum_jn rho(q,j,n) int3(p,j,n) K(e_p - e_j + Om_n, e_q - e_j + Om_n)
    shift = (om[None, :] - eo[:, None]).ravel()
    _srg_pair_sum(sig, eps, shift, int3.reshape(norb, -1),
                  np.ascontiguousarray(rho[:, o, :]).reshape(norb, -1), eta, workers)

    # part 2: int4(q,i,j,a) = sum_n -rho(q,j,n) g1[(i,a),n] + rho(j,q,n) g2[(i,a),n]
    int4 = (-(np.ascontiguousarray(rho[:, o, :]).reshape(norb * nO, ns) @ g1.T)
            + np.ascontiguousarray(rho[o, :, :].transpose(1, 0, 2)).reshape(norb * nO, ns) @ g2.T)
    int4 = int4.reshape(norb, nO, nO, nV).transpose(0, 2, 1, 3)
    bx_ija += wd * int4
    if wx:
        by_ija += wx * int4
    del int4

    # part 3: int3(p,b,n) = sum_ia (wd <pi|ab> + wx <pi|ba>) g1 + (wd <pa|ib> + wx <pa|bi>) g2
    n3 = wd * eri[:, o, v, v].transpose(0, 3, 1, 2)
    n3p = wd * eri[:, v, o, v].transpose(0, 3, 2, 1)
    if wx:
        n3 += wx * eri[:, o, v, v].transpose(0, 2, 1, 3)
        n3p += wx * eri[:, v, v, o].transpose(0, 2, 3, 1)
    int3 = n3.reshape(norb * nV, -1) @ g1 + n3p.reshape(norb * nV, -1) @ g2
    del n3, n3p
    # Sigma_pq += sum_bn rho(b,q,n) int3(p,b,n) K(e_p - e_b - Om_n, e_q - e_b - Om_n)
    rho_bq = np.ascontiguousarray(rho[v, :, :].transpose(1, 0, 2))           # [q, b, n]
    shift = (-ev[:, None] - om[None, :]).ravel()
    _srg_pair_sum(sig, eps, shift, int3.reshape(norb, -1), rho_bq.reshape(norb, -1), eta, workers)
    del int3

    # part 4: int4(q,i,a,b) = sum_n -rho(b,q,n) g1[(i,a),n] + rho(q,b,n) g2[(i,a),n]
    int4 = (-(rho_bq.reshape(norb * nV, ns) @ g1.T)
            + np.ascontiguousarray(rho[:, v, :]).reshape(norb * nV, ns) @ g2.T)
    int4 = int4.reshape(norb, nV, nO, nV).transpose(0, 2, 3, 1)
    bx_iab += wd * int4
    if wx:
        by_iab += wx * int4


def _pp_srg(sig, eps, eri, nO, ee, hh, w, eta, workers, bx_ija, bx_iab):
    """One pp spin channel of R_Parquet_self_energy_SRG_sp ('singlet/triplet
    pp part 1-4'): parts 1 and 3 into sig, parts 2 and 4 into the B factors
    of the (i, j, a) and (i, a, b) families."""
    om_ee, rho_ee = ee
    om_hh, rho_hh = hh
    norb = len(eps)
    nV = norb - nO
    o, v = slice(0, nO), slice(nO, norb)
    eo, ev = eps[o], eps[v]
    n_ee, n_hh = len(om_ee), len(om_hh)
    eoo = eo[:, None] + eo[None, :]
    evv = ev[:, None] + ev[None, :]
    # gee_ij = rho^ee(i,j,n) R(Om^ee_n - e_i - e_j), ghh_ij = rho^hh(i,j,n) R(e_i + e_j - Om^hh_n),
    # ghh_ab = rho^hh(a,b,n) R(Om^hh_n - e_a - e_b)
    gee_ij = (rho_ee[o, o, :] * _srg_inverse(om_ee - eo[:, None, None] - eo[None, :, None], eta)
              ).reshape(nO * nO, n_ee)
    ghh_ij = (rho_hh[o, o, :] * _srg_inverse(eoo[:, :, None] - om_hh, eta)).reshape(nO * nO, n_hh)
    ghh_ab = rho_hh[v, v, :] * _srg_inverse(om_hh - ev[:, None, None] - ev[None, :, None], eta)

    # part 1: int3(p,k,n) = -w sum_ij <pk|ij> gee_ij - w sum_ab <pk|ab> rho^ee(a,b,n) R(e_a + e_b - Om^ee_n)
    # part 4: int4(q,i,a,b) = sum_n rho^hh(q,i,n) ghh_ab + rho^ee(q,i,n) rho^ee(a,b,n) R(e_a + e_b - Om^ee_n)
    # both read the 2p amplitudes over the virtual pairs, in blocks of n.
    int3 = eri[:, o, o, o].reshape(norb * nO, nO * nO) @ gee_ij
    eri_ovv = eri[:, o, v, v].reshape(norb * nO, nV * nV)
    int4 = np.ascontiguousarray(rho_hh[:, o, :]).reshape(norb * nO, n_hh) @ ghh_ab.reshape(nV * nV, n_hh).T
    step = max(1, PARQUET_STATIC_PP_BLOCK_ELEMS // max(1, nV * nV))
    for n0 in range(0, n_ee, step):
        n1 = min(n0 + step, n_ee)
        g = (rho_ee[v, v, n0:n1] * _srg_inverse(evv[:, :, None] - om_ee[n0:n1], eta)).reshape(nV * nV, n1 - n0)
        int3[:, n0:n1] += eri_ovv @ g
        int4 += np.ascontiguousarray(rho_ee[:, o, n0:n1]).reshape(norb * nO, n1 - n0) @ g.T
        del g
    int3 *= -w
    bx_iab += w * int4.reshape(norb, nO, nV, nV)
    del int4, eri_ovv
    # Sigma_pq += sum_kn rho^ee(q,k,n) int3(p,k,n) K(e_p + e_k - Om^ee_n, e_q + e_k - Om^ee_n)
    shift = (eo[:, None] - om_ee[None, :]).ravel()
    _srg_pair_sum(sig, eps, shift, int3.reshape(norb, -1),
                  np.ascontiguousarray(rho_ee[:, o, :]).reshape(norb, -1), eta, workers)
    del int3

    # part 2: int4(q,i,j,a) = sum_n rho^ee(q,a,n) gee_ij + rho^hh(q,a,n) ghh_ij
    int4 = np.matmul(rho_ee[:, v, :], gee_ij.T) + np.matmul(rho_hh[:, v, :], ghh_ij.T)   # [q, a, (i,j)]
    bx_ija -= w * int4.reshape(norb, nV, nO, nO).transpose(0, 2, 3, 1)
    del int4

    # part 3: int3(p,c,n) = w sum_ij <pc|ij> ghh_ij + w sum_ab <pc|ab> ghh_ab
    int3 = (eri[:, v, o, o].reshape(norb * nV, nO * nO) @ ghh_ij).reshape(norb, nV, n_hh)
    int3 += _vvv_contract(eri, ghh_ab, nO)
    int3 *= w
    # Sigma_pq += sum_cn rho^hh(q,c,n) int3(p,c,n) K(e_p + e_c - Om^hh_n, e_q + e_c - Om^hh_n)
    shift = (ev[:, None] - om_hh[None, :]).ravel()
    _srg_pair_sum(sig, eps, shift, int3.reshape(norb, -1),
                  np.ascontiguousarray(rho_hh[:, v, :]).reshape(norb, -1), eta, workers)


def _vvv_contract(eri, g_vv, nO):
    """out[p, c, n] = sum_ab <pc|ab> g_vv[a, b, n] over virtual c, a, b: one
    GEMM on the whole eri (the occupied rows padded with zeros) when it is
    C-contiguous, else one per p."""
    norb = eri.shape[0]
    nV = norb - nO
    n = g_vv.shape[-1]
    if eri.flags.c_contiguous:
        g = np.zeros((norb, norb, n))
        g[nO:, nO:] = g_vv
        full = eri.reshape(norb * norb, norb * norb) @ g.reshape(norb * norb, n)
        return full.reshape(norb, norb, n)[:, nO:]
    g = g_vv.reshape(nV * nV, n)
    out = np.empty((norb, nV, n))
    for p in range(norb):
        out[p] = eri[p, nO:, nO:, nO:].reshape(nV, nV * nV) @ g
    return out


# --------------------------------------------------------------------------
# eta form
# --------------------------------------------------------------------------

def _eh_eta(sig, eps, eri, nO, chan, weights, lor, l_ija, l_iab):
    """One eh spin channel of R_Parquet_self_energy_eta: its three 3h2p and
    three 3p2h terms, each L(first) L(second) with L(d) = d / (d^2 + eta^2)."""
    om, rho = chan
    norb = len(eps)
    nV = norb - nO
    o, v = slice(0, nO), slice(nO, norb)
    eo, ev = eps[o], eps[v]
    wd, wx = weights
    ns = len(om)
    dia = ev[None, :] - eo[:, None]
    # g2[(i,a), n] = rho(a,i,n) L(e_a - e_i + Om_n); rho_ia[(i,a), n] = rho(i,a,n)
    g2 = (rho[v, o, :].transpose(1, 0, 2) * lor(dia[:, :, None] + om)).reshape(nO * nV, ns)
    rho_ia = np.ascontiguousarray(rho[o, v, :]).reshape(nO * nV, ns)

    # 3h2p. n1[p,j,i,a] = wd <pa|ij> + wx <pa|ji>, n1p[p,j,i,a] = wd <pi|aj> + wx <pi|ja>
    n1 = wd * eri[:, v, o, o].transpose(0, 3, 2, 1)
    n1p = wd * eri[:, o, v, o].transpose(0, 3, 1, 2)
    if wx:
        n1 = n1 + wx * eri[:, v, o, o].transpose(0, 2, 3, 1)
        n1p = n1p + wx * eri[:, o, o, v].transpose(0, 2, 1, 3)
    nl1 = n1 * l_ija.transpose(0, 2, 1, 3)                  # [p,j,i,a] L(e_p - e_i - e_j + e_a)
    del n1
    # terms 1 + 2: sum_jn rho(q,j,n) L(e_p - e_j + Om_n) [sum_ia nl1 rho(i,a,n) + n1p g2]
    m = (nl1.reshape(norb * nO, -1) @ rho_ia + n1p.reshape(norb * nO, -1) @ g2).reshape(norb, nO, ns)
    del n1p
    m *= lor(eps[:, None, None] - eo[None, :, None] + om[None, None, :])
    sig += m.reshape(norb, -1) @ np.ascontiguousarray(rho[:, o, :]).reshape(norb, -1).T
    del m
    # term 3: sum_ija nl1[p,j,i,a] sum_n g2[(i,a),n] rho(j,q,n)
    w3 = (g2 @ np.ascontiguousarray(rho[o, :, :]).reshape(nO * norb, ns).T).reshape(nO, nV, nO, norb)
    sig += nl1.transpose(0, 2, 3, 1).reshape(norb, -1) @ w3.reshape(-1, norb)
    del nl1, w3

    # 3p2h. n3[p,b,i,a] = wd <pi|ab> + wx <pi|ba>, n3p[p,b,i,a] = wd <pa|ib> + wx <pa|bi>
    n3 = wd * eri[:, o, v, v].transpose(0, 3, 1, 2)
    n3p = wd * eri[:, v, o, v].transpose(0, 3, 2, 1)
    if wx:
        n3 += wx * eri[:, o, v, v].transpose(0, 2, 1, 3)
        n3p += wx * eri[:, v, v, o].transpose(0, 2, 3, 1)
    nl3 = n3 * l_iab.transpose(0, 3, 1, 2)                  # [p,b,i,a] L(e_p + e_i - e_a - e_b)
    del n3
    # terms 4 + 5: sum_bn rho(b,q,n) L(e_p - e_b - Om_n) [-sum_ia nl3 rho(i,a,n) + n3p g2]
    m = (n3p.reshape(norb * nV, -1) @ g2 - nl3.reshape(norb * nV, -1) @ rho_ia).reshape(norb, nV, ns)
    del n3p
    m *= lor(eps[:, None, None] - ev[None, :, None] - om[None, None, :])
    sig += m.reshape(norb, -1) @ np.ascontiguousarray(rho[v, :, :].transpose(1, 0, 2)).reshape(norb, -1).T
    del m
    # term 6: sum_iab nl3[p,b,i,a] sum_n g2[(i,a),n] rho(q,b,n)
    w6 = (g2 @ np.ascontiguousarray(rho[:, v, :]).reshape(norb * nV, ns).T).reshape(nO, nV, norb, nV)
    sig += (nl3.transpose(0, 2, 3, 1).reshape(norb, -1)
            @ w6.transpose(0, 1, 3, 2).reshape(-1, norb))


def _pp_eta(sig, eps, eri, nO, ee, hh, w, lor, l_ija, l_iab):
    """One pp spin channel of R_Parquet_self_energy_eta: the 4h1p and 3h2p
    terms (2p and 2h branch) of its first loop, the 4p1h and 3p2h terms of
    its second, each L(first) L(second)."""
    om_ee, rho_ee = ee
    om_hh, rho_hh = hh
    norb = len(eps)
    nV = norb - nO
    o, v = slice(0, nO), slice(nO, norb)
    eo, ev = eps[o], eps[v]
    n_ee, n_hh = len(om_ee), len(om_hh)
    # gee_ij = rho^ee(i,j,n) L(Om^ee_n - e_i - e_j), ghh_ab = rho^hh(a,b,n) L(Om^hh_n - e_a - e_b)
    gee_ij = (rho_ee[o, o, :] * lor(om_ee - eo[:, None, None] - eo[None, :, None])).reshape(nO * nO, n_ee)
    ghh_ab = rho_hh[v, v, :] * lor(om_hh - ev[:, None, None] - ev[None, :, None])

    # el[p,k,a,b] = <pk|ab> L(e_p + e_k - e_a - e_b)
    el = eri[:, o, v, v] * l_iab
    # 4h1p + 3p2h (2p branch): sum_kn rho^ee(q,k,n) L(e_p + e_k - Om^ee_n)
    #     [-w sum_ij <pk|ij> gee_ij + w sum_ab el[p,k,a,b] rho^ee(a,b,n)]
    m = -w * (eri[:, o, o, o].reshape(norb * nO, nO * nO) @ gee_ij)
    el_flat = el.reshape(norb * nO, nV * nV)
    step = max(1, PARQUET_STATIC_PP_BLOCK_ELEMS // max(1, nV * nV))
    for n0 in range(0, n_ee, step):
        n1 = min(n0 + step, n_ee)
        m[:, n0:n1] += w * (el_flat @ np.ascontiguousarray(rho_ee[v, v, n0:n1]).reshape(nV * nV, n1 - n0))
    m = m.reshape(norb, nO, n_ee)
    m *= lor(eps[:, None, None] + eo[None, :, None] - om_ee[None, None, :])
    sig += m.reshape(norb, -1) @ np.ascontiguousarray(rho_ee[:, o, :]).reshape(norb, -1).T
    del m, el_flat
    # 3h2p (2p branch): sum_cij -w <pc|ij> L(e_p + e_c - e_i - e_j) sum_n gee_ij rho^ee(q,c,n)
    lc = l_ija.transpose(0, 3, 1, 2)                         # [p,c,i,j] L(e_p + e_c - e_i - e_j)
    vl = eri[:, v, o, o] * lc                                # [p,c,i,j]
    wb = np.matmul(rho_ee[:, v, :], gee_ij.T)                # [q, c, (i,j)]
    sig -= w * (vl.reshape(norb, -1) @ wb.reshape(norb, -1).T)
    del wb
    # 3h2p (2h branch) + 4p1h: sum_cn rho^hh(q,c,n) L(e_p + e_c - Om^hh_n)
    #     [-w sum_ij vl[p,c,i,j] rho^hh(i,j,n) + w sum_ab <pc|ab> ghh_ab]
    m = (-w * (vl.reshape(norb * nV, nO * nO) @ np.ascontiguousarray(rho_hh[o, o, :]).reshape(nO * nO, n_hh))
         ).reshape(norb, nV, n_hh)
    del vl
    m += w * _vvv_contract(eri, ghh_ab, nO)
    m *= lor(eps[:, None, None] + ev[None, :, None] - om_hh[None, None, :])
    sig += m.reshape(norb, -1) @ np.ascontiguousarray(rho_hh[:, v, :]).reshape(norb, -1).T
    del m
    # 3p2h (2h branch): sum_kab w el[p,k,a,b] sum_n ghh_ab rho^hh(q,k,n)
    we = (ghh_ab.reshape(nV * nV, n_hh) @ np.ascontiguousarray(rho_hh[:, o, :]).reshape(norb * nO, n_hh).T
          ).reshape(nV, nV, norb, nO)
    sig += w * (el.reshape(norb, -1) @ we.transpose(3, 0, 1, 2).reshape(-1, norb))


# --------------------------------------------------------------------------
# inputs and flags
# --------------------------------------------------------------------------

def _checked(eta, eps, eri, nO, spectra):
    """(eta, eps, eri, nO, {channel: (om, rho)}) as float arrays and an int,
    or ValueError for an input that cannot be Quack's: a negative or
    non-finite eta, shapes that do not match norb, nO and the pair counts."""
    eta = float(eta)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError(f'eta={eta!r}: the regularization is a finite eta >= 0')
    eps = np.asarray(eps, dtype=float)
    if eps.ndim != 1:
        raise ValueError(f'eps has shape {eps.shape}; expected (norb,)')
    norb = len(eps)
    eri = np.asarray(eri, dtype=float)
    if eri.shape != (norb,) * 4:
        raise ValueError(f'eri has shape {eri.shape}; expected {(norb,) * 4}')
    if int(nO) != nO or not 0 < nO < norb:
        raise ValueError(f'nO={nO!r}: expected an integer with 0 < nO < norb = {norb}')
    nO = int(nO)
    nV = norb - nO
    if not isinstance(spectra, RestrictedSpectra):
        raise ValueError('spectra must be a RestrictedSpectra')
    counts = {'eh_singlet': nO * nV, 'eh_triplet': nO * nV,
              'ee_singlet': pp_pair_count(nV, 'singlet'), 'ee_triplet': pp_pair_count(nV, 'triplet'),
              'hh_singlet': pp_pair_count(nO, 'singlet'), 'hh_triplet': pp_pair_count(nO, 'triplet')}
    chan = {}
    for name, count in counts.items():
        om, rho = getattr(spectra, name)
        om = np.asarray(om, dtype=float)
        rho = np.asarray(rho, dtype=float)
        if om.shape != (count,) or rho.shape != (norb, norb, count):
            raise ValueError(f'spectra.{name}: om {om.shape}, rho {rho.shape}; expected '
                             f'({count},) and {(norb, norb, count)}')
        chan[name] = (om, rho)
    return eta, eps, eri, nO, chan


def _flag_not_finite(sig, where):
    """Warn and raise 'parquet_static_sigma_not_finite' if sig holds a NaN or inf."""
    bad = ~np.isfinite(sig)
    if bad.any():
        flag(FLAG_STATIC_SIGMA_NOT_FINITE,
             f'{where}: {int(bad.sum())} of {sig.size} elements of the static parquet '
             f'self-energy are not finite (an energy, a spectrum or an integral that is '
             f'not finite, an overflow, or a zero denominator of the eta form at a '
             f'broadening whose square is 0: eta = 0, its unregularized limit, or below '
             f'about 1.6e-162 Ha); returned as computed',
             ParquetNotFinite, stacklevel=3)
