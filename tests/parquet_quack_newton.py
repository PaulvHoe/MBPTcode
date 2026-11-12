"""Quack's R_Parquet_QP_graph, for the parquet tests that compare with Quack.

Quack (the paper-era tree, git e3528a40) and the paper's numbers (Marie and
Loos, J. Chem. Phys. 163, 194115 (2025)) solve the quasiparticle equation
w = eHF_p + Sigma_pp(w) of FLEX, osPA and evPA by Newton from the linearized
energy eQPlin_p,

    f = w - eHF_p - Sigma_pp(w),   Z = 1 / (1 - dSigma_pp/dw),   w <- w - Z f,

while |f| > NEWTON_TOL and fewer than NEWTON_MAX_IT steps ran (R_evParquet
for the closed shell; G_evParquet has no search and stops unless the
equation is linearized). The parquet drivers solve it by the house's root
finder instead (qp_solver='find_qp_root', the default); `quack_newton` is
Quack's search as a qp_solver function of both drivers
(src.SingleReference.Parquet.options.checked_qp_solver), so that a test, a
comparison script or a recipe that wants Quack's numbers to rounding passes
qp_solver=quack_newton. Its arithmetic is R_Parquet_QP_graph's, as the
drivers ran it before it left the package:

- the energy kept is the iterate after the last step, and z is
  1/(1 - dSigma/dw) at the iterate before it;
- an orbital still running after NEWTON_MAX_IT steps keeps eQPlin and the Z
  handed in (the diagonal Z at eOld), even where the last step met the
  tolerance, and is not converged;
- a NaN residual ends the loop as a convergence would; the NaN iterate is
  kept, not converged.

The drivers flag what it returns as for any qp_solver function: a finite
energy not converged ('parquet_qp_not_converged'), an energy not finite
('parquet_qp_not_finite'), a Z outside (0, 1] ('z_outside').

Where the search wanders between the dense poles of a weakly regularized
self-energy (s_1b near 1e8; |dSigma/dw| of 1e2 to 1e4 along the path) it is
chaotic: a difference in the last bit of its start grows to order one within
some twenty steps, and Quack's own root, step count and failure move with one
ulp of eQPlin or eOld. The same holds for the Z of a root beside a pole,
taken at the iterate before the last step. Such orbitals match Quack only as
well as Quack matches itself: with this search the closed-shell driver
reproduces Quack's printed table of ammonia/6-31G osPA at s_1b = 1e8 to its
printed digits (5e-7 eV) for every orbital but two high virtuals, whose
roots lie 44.7 and 9.4 eV from Quack's (Z of 0.0008 and 0.35 in the port,
0.013 and 0.029 in Quack) and move, in the port, by tens of eV when eQPlin
moves by one ulp.

AGAINST THE DRIVERS' DEFAULT, the house's root finder, as measured on the
drivers:

- The same root. Where both reach it they agree to this search's tolerance
  (its residual |f| reaches 4.5e-7 Ha, the finder's 5e-12): in FLEX and in
  osPA with s_2b 1 and two-body DIIS 5 on water and ammonia 6-31G, the
  principal IP to 1e-11 eV, and at s_1b = 50 every orbital to 1.3e-5 eV;
  the principal IPs of docs/methods/parquet.md's paper-table recipes at
  small size (FLEX TDA and RPA at s_1b = 1e8 on Ne, water and ammonia in
  cc-pVDZ, Table II's osPA on Ne/cc-pVDZ, Table III's osPA on
  water/cc-pVDZ) to 1e-12 eV.
- Another root. Where this search wanders between poles (weak
  regularization, high virtuals) or stops at a root of smaller weight, the
  finder keeps another (water/6-31G osPA at the restricted driver's
  defaults: orbital 8 1.25 eV apart, where this search was capped and kept
  eQPlin). Beside the pole of an eh instability (H2/6-31G 1e-6 Angstrom
  before its RHF triplet instability, FLEX with the full BSEs, s_1b = 50)
  this search puts the HOMO at -324 eV with Z = 0.5, the finder at +22.9 eV
  with Z = 0.0012.
- evPA. Following its roots, the finder converges ammonia/6-31G (TDA, s_2b
  1, s_1b 50) in this search's iteration count, every orbital within 1e-5 eV
  of it (the N 1s root this search reaches has Z 0.58, where re-deciding by
  weight cycled through roots of 0.70, 0.83 and 0.66), and water to
  3e-6 eV. On water/cc-pVDZ at the paper's Table III settings it converges
  in 7 one-body iterations with one-body and two-body DIIS 5, this search in
  12 (the HOMOs 0.7 meV apart, within tau_1b), and in 11 with two-body DIIS
  alone, where this search had not converged after 50.
- Spin pairs. Without run_parquet_so's pairing this search can take an
  RHF's alpha and beta copies, equal only to roundoff, to different roots:
  water/6-31G evPA at s_1b = 1e8 (s_2b 1, damping 0.75) split them by
  7e-14, 0.15, 5.3 and 1.4 eV in its first four iterations, the HOMO by
  3.6 meV at the fourth; the finder kept them within 1.4e-13 eV.
- Cost. This search takes 1 to 52 steps per orbital (64 where it is
  capped) on water and ammonia 6-31G and water/aug-cc-pVDZ, and 2 to 37 on
  the spin-orbital equations of water and ammonia 6-31G; the finder 150 to
  14 000 evaluations without a root to follow, and about as many with one.
  Per one-body step: on water/aug-cc-pVDZ (FLEX TDA) 0.05 s against 7.7 s
  at s_1b = 50 and 33 s at 1e8; three evPA steps there (s_1b = 50,
  s_2b = 1, two-body DIIS 5, two threads) 0.1 s against 14 s, the two
  steps that follow their roots taking more evaluations than the first
  (45 000 and 49 000 against 32 000); at nso = 26 to 30 in
  spin orbitals 0.1 to 2 s against 6 to 36 s; on N2/aug-cc-pVTZ every
  orbital in 3 s; on water/aug-cc-pVDZ in spin orbitals (nso 82, one
  thread) all 82 in 19 s. Whole runs: docs/methods/parquet.md's first
  example 1.2 s against 5.0 s and its open-shell example 0.2 s against
  8.5 s; run_parquet_so's osPA at its defaults on water and ammonia 6-31G,
  whose loop diverges, 3.6 and 8.7 s against 173 and 200 s.

Not a test module: pytest collects nothing from it.
"""
import numpy as np

#: R_Parquet_QP_graph's thresh (Hartree): the search stops once
#: |w - eHF_p - Sigma_pp(w)|, taken before the step, is at most this.
NEWTON_TOL = 1e-6
#: R_Parquet_QP_graph's maxIt: after this many steps the orbital keeps its
#: linearized energy and Z, even when the last step met the tolerance.
NEWTON_MAX_IT = 64


def quack_newton(sigma, eps_hf, eqp_lin, z_lin, orbitals):
    """(eqp, z, converged): R_Parquet_QP_graph on each of `orbitals`, a
    qp_solver function of the parquet drivers (module docstring).

    sigma(p, w) is the driver's (Sigma_pp(w), dSigma_pp/dw), two numpy
    float64; eps_hf, eqp_lin and z_lin are indexed by orbital; the arrays
    returned are eqp_lin and z_lin where an orbital is not solved, or capped.
    """
    eqp, z = np.array(eqp_lin, dtype=float), np.array(z_lin, dtype=float)
    converged = np.zeros(len(eqp), dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for p in orbitals:
            w, f, df, nit = np.float64(eqp_lin[p]), np.float64(1.0), np.float64(np.nan), 0
            while abs(f) > NEWTON_TOL and nit < NEWTON_MAX_IT:
                nit += 1
                sig, dsig = sigma(p, w)
                f = w - eps_hf[p] - sig
                df = 1.0 / (1.0 - dsig)
                w = w - df * f
            if nit < NEWTON_MAX_IT:
                eqp[p], z[p] = w, df
                converged[p] = abs(f) <= NEWTON_TOL
    return eqp, z, converged
