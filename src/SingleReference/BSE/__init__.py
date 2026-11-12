"""Bethe-Salpeter equation beyond the static kernel.

bse_upfolded.py carries the frequency-free (upfolded) dynamical BSE of
Bintrim and Berkelbach, J. Chem. Phys. 156, 044114 (2022): the singles space
is augmented by the doubles that the dynamical kernel resums, so the
eigenvalue problem is linear and the frequency dependence is recovered
exactly on downfolding.

solve.py is the static BSE@GW: the front end `solve_bse` and its drivers,
on the Casida eigensolvers of LinearResponse/ (casida.py, dense; davidson.py,
matrix-free). quasi_boson_bse.py is the dense quasi-boson route.

second_order_bse.py is BSE(2) of van Hoegaerden and Foerster, the second-order
kernel of LinearResponse/second_order_kernel.py on the bare first-order BSE,
with its dBSE(2) correction; second_order_qp.py runs the quasiparticle
sequences of that paper's AMS runs, which put their energies on its diagonal.
"""
