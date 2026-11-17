# Warnings, flags and refusals

The code answers trouble in one of two ways, depending on where it comes from.

**A wrong input is refused.** A request that cannot give a meaningful answer
raises an error before any work is done, and the message names what is
wrong. Examples: an ADC(4) or higher calculation that does not say which
construction to use, an ISDF grid the shipped radii table has no row for,
EOM-CCSD from the coupled-cluster package (use pyscf's), a property that needs
an ingredient the surface does not provide (`MissingIngredient`), or two
different sets of numerical settings in one calculation.

**A numerical shortfall is warned about and flagged.** When a solver stops
short (an iteration runs out of cycles, a root is not found, a weight comes
out unphysical), the code keeps its last result, issues a Python warning, and
names the event in what it returns. The run goes on. Read the flags before
you trust a number: a flagged number is the best the solver reached, not a
converged one.

Some events are handed back by the solver itself, in the place the table
below gives (`info['converged']`, `out['no_root']`, ...). The others are
*named flags* (`src.Base.numerical_flags`): the solver warns and raises the
flag, and the function that returns the result collects every flag raised
inside it, on any thread or MPI rank, into one list of
`{'flag': name, 'message': text}` entries:

- `calc_qp_energy(..., out=out)`: `out['numerical_flags']`;
- an ADC solver: `solver.last_result['numerical_flags']` after `solve()`
  (every flag of this solver so far, since later solves reuse its
  amplitudes);
- an excited-state chain (`ExcitedStateChain`) and `DenseBSESurface`: the
  `numerical_flag_warnings` of their vertical-excitation and emission records,
  with the geometry each flag was first raised at; the dense surfaces' gradients
  also return them in their diagnostics, `diagnostics['numerical_flags']`;
- a photophysics stage: the stage record's `numerical_flags`, as
  `{flag: [messages]}`;
- a parquet run (`SingleReference.Parquet`: `run_parquet`, `ev_parquet`,
  `qs_parquet` and the spin-orbital `run_parquet_so`, `ev_parquet_so`,
  `qs_parquet_so`): `result.numerical_flags`; its quasiparticle root
  searches on their own (`DiagonalSelfEnergy.solve_qp_scan`, the
  spin-orbital `solve_qp_scan_so`) also write them to
  `out['numerical_flags']`. With `qp_solver='find_qp_root'` they hold the
  root finder's own `'no_root'` and `'z_outside'` (the rows of every route
  below), for both spin orbitals of a spin pair (an RHF through
  `run_parquet_so`);
- the BSE(2) driver `second_order_bse` and its quasiparticle sequences
  `qp_sequence`: their result's `numerical_flags` (the sequence's own are also
  in `res['qp']['numerical_flags']`);
- the soft-mode split `soft_mode_triplets` and its quasiparticle driver
  `soft_mode_qp`: their result's `numerical_flags` (the split's are also in
  `res['triplets']['numerical_flags']`).

A flag raised outside all of these, by a solver called on its own, is
warned about and nothing more.

## The flags

| where | what stopped short | warning | flag |
|---|---|---|---|
| quasiparticle equation, every route | no root in the widest scan window | `QPNoRoot` | the root is NaN, named flag `'no_root'`; `out['no_root']` of the root finder |
| quasiparticle equation, every route | the root kept has pole strength Z outside (0, 1]; the root is kept as solved | `QPWeightOutside` | named flag `'z_outside'`; `out['z_outside']` of the root finder; on a gradient chain `qp_z_outside` |
| excited-state chain on the analytic continuation (`--residues pade`) | the Pade continuation of a quasiparticle-set member failed numerically; it takes the correction of its nearest solved member (a failed HOMO or LUMO is refused) | `QPContinuationFailed` | named flag `'qp_continuation_failed'`; the chain's `qp_diagnostics['continuation_failed']` |
| quasiparticle equation, a root followed from a seed | the largest pole strength has moved to another root | `RuntimeWarning` | the root is kept on its branch; `QPqb.branch_log`, on a gradient chain `qp_branch_warnings` |
| quasi-boson BSE, the set decided at the reference | an orbital's quasiparticle root has pole strength Z of 0.5 or less | `QPWindowMeanField` | named flag `'qp_window_mean_field'`; the BSE diagonal keeps that orbital's mean-field energy |
| evGW | the loop ran out of cycles | `EvGWNotConverged` | named flag `'evgw_not_converged'`; `info['converged']` False |
| qsGW | the loop ran out of cycles before a fixed point | `QsGWNotConverged` | named flag `'qsgw_not_converged'`; `info['converged']` False; the spectrum is the last iterate |
| screening on the imaginary axis | 1 - chi0(i nu) is not positive definite, so Cholesky fails; solved by LU | `ScreeningNotPositive` | named flag `'screening_not_positive_definite'` |
| quasiparticle Hamiltonian, a group of more than 24 orbitals | the GMRES on its Newton equation stopped above its tolerance; the last iterate is kept and the Newton goes on (its stop is the exact residual) | `SubspaceGMRESShort` | named flag `'qp_subspace_gmres_not_converged'` |
| sum-over-poles vector fitting | the least squares failed; solved by another LAPACK driver (gelss, then gelsy) | `VectorFitFallback` | named flag `'vector_fit_lstsq_fallback'` |
| dRPA correlation energy (imaginary axis, space-time and periodic) | det(1 - chi0) is not positive at a frequency; log\|det\| is used, and the energy is not the dRPA one | `NonPositiveDielectric` | named flag `'nonpositive_det'`; with a `flags` dict passed, `flags['nonpositive_det']`, the frequencies (and transfers) where it happened |
| periodic THC GW | no tabulated minimax grid reaches the asked accuracy; the largest grid runs | `MinimaxGridShort` | with `return_diagnostics=True`, `diag['numerical_flags']['minimax_tau_target_missed']`, the accuracy reached |
| Davidson unit-vector seeds (`Solvers.davidson`, and the Casida / BSE guess in `LinearResponse.davidson_shared`) | the seed count, extended past a cut inside a cluster of degenerate diagonal entries, reached its cap; the seeds may split a degenerate set | `RuntimeWarning` | named flag `'davidson_guess_truncated'`; `counters['guess_truncated']`, which the Casida / BSE solve's `timings` carry as `'davidson_guess_truncated'` |
| Casida / BSE Davidson | a root above its residual tolerance, or a root an unsettled higher pair may lie below | `RuntimeWarning` | `info['davidson_flags']`: `'davidson_not_converged'`, `'davidson_unconfirmed'` |
| BSE(2) dynamical correction (`BSE.second_order_bse.dynamical_correction`) | the renormalization zeta = 1/(1 - slope) of a root is not positive | `RuntimeWarning` | named flag `'second_order_bse_zeta_nonpositive'`; the root's row is returned |
| BSE(2) quasiparticle sequences (`BSE.second_order_qp.quasiparticle_cycle`) | an orbital has no root of positive weight in AMS's one scan window; it keeps its HF energy, as AMS keeps it | `RuntimeWarning` | named flag `'qp_sequence_no_root'`; Z is NaN in the cycle's record |
| BSE(2) quasiparticle sequences (`BSE.second_order_qp.quasiparticle_cycle`) | the root kept for an orbital has Z <= 0.5: no quasiparticle, kept as AMS keeps it | `RuntimeWarning` | named flag `'qp_sequence_low_weight'`; the root is kept |
| BSE(2) quasiparticle sequences (`BSE.second_order_qp.quasiparticle_cycle`) | the root kept for an orbital has Z > 1: AMS's rule keeps the largest weight, one above 1 included | `RuntimeWarning` | named flag `'qp_sequence_weight_above_one'`; the root is kept |
| soft-mode split of a TDHF triplet instability (`LinearResponse.soft_mode.soft_mode_triplets`) | a soft mode's collective ground state has more than `SOFT_MODE_EDGE_WEIGHT_MAX` of its weight near the ends of its RHF->UHF path, or its first excitation moves by more than `SOFT_MODE_BOUNDARY_SHIFT_MAX` when the periodic path is cut by walls; the periodic levels are kept | `SoftModeEdgeWeight` | named flag `'soft_mode_edge_weight'`; the mode's `edge_weight` and `excitation_walls` in `res['soft']` |
| soft-mode split of a TDHF triplet instability (`LinearResponse.soft_mode.soft_mode_triplets`) | more than one set of soft triplet modes: each has its own collective model along its own path, the couplings between them are left out; the levels are kept | `SoftModesCoupled` | named flag `'soft_modes_coupled'` |
| orbital response: the CPHF / Z-vector solve of a relaxed density (`Base.pyscf.cphf`) and the gradient layer's multiplier solve (`gradients.multipliers`) | the residual stayed above its tolerance; the last iterate is kept | `CphfNotConverged` | named flag `'orbital_response_not_converged'`, also in a `flags` list the caller hands the multiplier solve |
| CC amplitudes and Lambda (also pyscf's CCSD under the CC density) | out of iterations | `CCNotConverged` | `converged` False, and the named flag `'cc_not_converged'` |
| an iteration inside an ADC route (the CCD/CCSD or Brueckner amplitudes of Faddeev- and CC-ADC(3), the ladder CCD, a FLEX Lanczos resolvent, the Lanczos inverse square root of the Faddeev metric) | out of iterations or steps | `ADCNotConverged` | `out['converged']`, `info['converged']` or `solve_qp(...)['converged']` False, and the named flags `'cc_amplitudes_not_converged'`, `'brueckner_not_converged'`, `'ladder_amplitudes_not_converged'`, `'flex_resolvent_not_settled'`, `'inverse_sqrt_not_settled'` |
| coupled cluster as an active-space solver (`CoupledClusterSolver`, the `'ccsdt'` / `'ccsdtq'` backends of `active_space.solve`) | the Hartree-Fock reference in the active space, or the amplitudes, out of cycles; the last iterate's energy is returned | `ActiveSpaceCCNotConverged` | `solver.last_result`: `reference_converged`, `converged`, and in `numerical_flags` the named flags `'active_space_reference_not_converged'`, `'active_space_cc_not_converged'` |
| EOM-CC | unconverged roots | `UnconvergedEOMRoots` | `EOMResult.converged`, and the named flag `'eom_roots_not_converged'` |
| EOM-CC | a state whose left eigenvector was not found, so it cannot be paired with its right one; its transition densities are zero, and G0W@CC drops its pole from Sigma_c | `UnpairedEOMStates` | `EOMResult.unpaired_states`, and the named flag `'eom_states_unpaired'` |
| EOM-CC, non-Hermitian solve | eigenvalues with an imaginary part above `EOM_IMAG_TOL`; the real parts are kept | `ComplexEOMEigenvalues` | `EOMResult.complex_eigenvalues`, and the named flag `'eom_complex_eigenvalues'` |
| ground-state relaxation | the walk stopped before converging | `UnconvergedRelaxation` | `mf.relaxation`: `relaxation_converged`, `opt_grad_max` |
| excited-state walk | the walk did not converge | `WalkNotRelaxed` | `'walk_not_converged'` |
| refreeze loop | the energy still moves by more than 3 meV after 4 passes | `RefreezeNotConverged` | `refreeze='not converged'`, `'refreeze_not_converged'` |
| state-specific solvation (`environments.solvation_loop`) | the loop ran out of cycles (its last cycle is returned), or the state model returned no number (the result is NaN) | `SolvationNotConverged` | `LoopResult.flags`, beside the state model's own flags of that cycle: `'solvation_loop_not_converged'`, `'solvation_loop_stopped'` |
| a scheme-B minimum (`photophysics.state_specific.walk_vs_vertical`) | G_eq of the state at a walk's minimum on the walk's surface and on the vertical's there differ by more than the bar (3 meV after a refreeze, else ten loop tolerances) | `RuntimeWarning` | `walk_vs_vertical_flag` in the record, and the stage pipeline's `'walk_vs_vertical'` |
| photophysics driver (`photophysics.solvated`) | no memory budget given (no `--max-memory`, SLURM allocation or `PYSCF_MAX_MEMORY`): each rank keeps pyscf's default, and a tiled force whose estimate exceeds it is refused | `RuntimeWarning` | named flag `'memory_budget_unknown'`; the record's `run_flags` |
| a scheme-B T1 stage on the full BSE (`photophysics.stages.tda_single_point`) | the triplet walk stopped at the full kernel's instability (`CasidaInstability`, `CasidaBreakdown`): T1 is the full-BSE single point at the declared Tamm-Dancoff T1 minimum, not a minimum of the full kernel | `TdaSinglePoint` | `'tda_minimum_single_point'`, and `tda_single_point` (why, where the walk stopped, the minimum's sha256) in the record |
| a force on a loosely converged mean field | the SCF stopped above its force tolerance | `RuntimeWarning` | `scf_flags` |
| an SCF of the photophysics chain | out of SCF cycles; the stage goes on | `SCFNotConverged` | the stage's `numerical_flags`: `'scf_not_converged'`, the count |
| Franck-Condon rate integral | the time integrand had not decayed; the rate is computed on the truncated density | `FCIntegrandNotDecayed` | the rate table's `numerical_flags`: `'fc_integrand_not_decayed'` |
| Herzberg-Teller dipole derivative by finite differences (`dipole_derivative`) | along a mode the dipole changes mostly evenly in the step, or a displaced dipole points against mu0 (a phase flip between the displaced solves, or a dipole far from linear); the slope is kept | `TransitionDipolePhase` | `TransitionDipole.suspect_modes` |
| ISC/RISC rate chain on a Duschinsky density (`photophysics.rate_chain`) | a density below 1e-13 of its band's peak, the real-time integral's roundoff floor (a deeply activated crossing); the rate is kept and is not resolved | `FCDensityAtRoundoff` | the chain's `numerical_flags`: `'fc_density_at_roundoff'` (and `'fc_integrand_not_decayed'` as above) |
| Duschinsky model from two Hessians (`duschinsky_model`) | the overlap of the two mode sets is not orthogonal to 2e-2 (axis switching the alignment did not remove); its nearest orthogonal matrix is used | `DuschinskyNotOrthogonal` | `DuschinskyModel.orthogonality` |
| the photophysics rate table | the ISDF-vs-DF acceptance check failed at a stage the table reads; the table is computed | `AcceptanceCheckFailed` | the rate table's `numerical_flags`, under that stage: `'isdf_df_check_failed'` |
| ISDF points pivoted for the AO-product fit (`ao_product_fit.pivoted_points`) | the parent grid's Gram matrix has a numerical rank below the asked point count; the points up to that rank are kept | `ISDFPivotShort` | named flag `'isdf_pivot_rank_short'` |
| the tight SCF of a dense GW-BSE surface (`tight_rhf`) | out of SCF cycles; the last iterate is kept | `RuntimeWarning` | named flag `'scf_not_converged'`, also in `mf.flags` |
| finite-difference Hessian (`surface_hessian`, the mode Hessian) | the two halves differ by more than 1e-3 of the largest force constant (forces not reproducible, or a grid too coarse for the step); the Hessian is kept symmetrized | `HessianAsymmetric` | `info['hessian_asymmetric']`, `'hessian_asymmetric'` in `info['flags']`; a saddle check's record and its stage's `numerical_flags`: `'hessian_asymmetric'` |
| fragment diabats: the localization response | MINRES stopped above its tolerance; the last iterate is kept | `RuntimeWarning` | `'localization_response_not_converged'` |
| fragment diabats: a diabat's response | the solve stopped above its tolerance; the last iterate is kept | `RuntimeWarning` | `'diabat_response_not_converged'` |
| quasiparticle equation, a root followed from a seed in the finite-difference reference of the quasi-boson route (`quasi_boson.qp_energy_general`) | the followed root has pole strength Z at or below 0.5; it is kept on its branch | `QPLowWeightRoot` | named flag `'qp_followed_root_low_weight'` |
| upfolded BSE (`BSE.bse_upfolded`) | a root with a significant imaginary part (the dynamical BSE has no real root there): returned as NaN | `RuntimeWarning` | `'upfolded_bse_complex_roots'` in the `flags` list the caller hands in |
| upfolded BSE (`BSE.bse_upfolded`) | a root with Re <= 0 is left out of the returned excitation energies (the dense eigenvalues, or the Ritz values of the Davidson's final subspace) | `RuntimeWarning` | `'upfolded_bse_dropped_roots'` in the `flags` list the caller hands in |
| upfolded BSE, Davidson | out of cycles, or a linearly dependent correction above tolerance; the Ritz pairs are kept | `RuntimeWarning` | `'upfolded_bse_not_converged'` in the `flags` list the caller hands in; `refuse_unconverged=True` raises instead |
| excitation ADC(2) as a state model and its force (`gradients.adc.ee_state_model`, `ee_gradient`) | a root not converged (kept as solved); a root index the solve did not return (NaN energy, density and force, which a walk rejects); a followed root that overlaps the roots solved by less than 0.5, or two roots within 0.2 of each other in overlap (the best overlap is taken) | `FollowingWarning` | `info['flags']`: `'adc_roots_not_converged'`, `'adc_root_missing'`, `'root_following_lost'`, `'root_following_mixed'` |
| excitation ADC(2) force | the force re-solved the roots and its root is not the call's: under `follow='index'` its transition amplitude overlaps the call's root by less than 0.5; at the call's own field its omega differs from the call's | `FollowingWarning` | the force's `info['flags']`: `'force_root_order_differs'`, `'force_omega_mismatch'` |
| folded excitation ADC(2) (`eeADC.ee_fold.solve_folded`), each event's solve carrying on | a root, level or branch above its tolerance (`'root_not_converged'`, `'level_not_converged'`, `'branch_root_not_converged'`, `'roots_not_converged'`); model roots or a model branch not converged (`'model_roots_not_converged'`, `'model_branch_unsolved'`); a vector or level whose overlap with its previous one fell below 0.5, kept (`'root_crossing'`, `'level_crossing'`); a level or block of branches not degenerate in A_eff, re-solved per root (`'level_split'`, `'branches_split'`); two roots on one, the copy dropped (`'duplicate_root_dropped'`); fewer roots than asked (`'roots_missing'`); roots reaching the lowest doubles energy, solved per root (`'block_reaches_doubles'`), or lying above it, not checked by the count (`'roots_above_doubles_unchecked'`); a state below the roots that a warm solve's start lacked and only the full check's random-vector walk found (`'fold_root_crossed'`: earlier warm solves may have followed another state; the ADC(2) state model's `confirm` raises it at a walk's converged geometry, and the walk then ends not converged) | `RuntimeWarning` | `FoldResult.flags`, which the ADC(2) state model copies into `info['flags']`; at a walk's converged geometry `info['walk_flags']` |
| fragment partition of a singles operator, GW-BSE or ADC(2) (`fragments.partition`) | a block Davidson short of its tolerance (`'block_eigenpairs_not_converged'`); a Q-space lowest-root pass that stopped above tolerance (`'q_lowest_unconverged'`); the pole guard's Newton (`'pole_guard_newton_not_converged'`); the resolvent's conjugate gradients (`'resolvent_not_converged'`); the last iterate is kept each time | `RuntimeWarning` | `FragmentPartition.guard_flags` (`{name: detail}`) for the partition's solves, which every diabat gradient (`DiabaticGradient.gradient`, `element`, `root_gradient`) carries in its `flags`, each also a named flag |
| parquet two-body loop (`Parquet.restricted`: `ev_parquet`, `qs_parquet`, `run_parquet`; `Parquet.spin_orbital`: `ev_parquet_so`, `qs_parquet_so`, `run_parquet_so`) | the loop ran `max_it_2b` iterations (Quack counts n_it == max_it_2b as failed, also when the last one met `conv_2b`); the run goes on with the last vertices. Without two-body DIIS a damped loop can oscillate from s_2b of about 10 (water/6-31G at s_2b = 10: a change of 0.26 after 400 iterations), so `run_parquet`'s defaults (damping 0.75, no DIIS, s_2b = 50) leave osPA of water/6-31G at a change of 5.6 after 200 iterations, and `run_parquet_so`'s flex, os and ev loops are undamped by default (damping 0, G_evParquet's), which diverges at s_2b = 50 (osPA of water, ammonia and OH in 6-31G: a change of 1e122 to 1e128 after 200 iterations). Raise `max_it_2b`, turn on two-body DIIS (`max_diis_2b=5`), damp (`damping=0.75`), or lower `eta_2b` | `ParquetNotConverged` | named flag `'parquet_two_body_not_converged'`; that one-body iteration's `converged_2b` False |
| parquet one-body loop of evPA and qsPA (the same drivers, schemes `'ev'` and `'qs'`) | the loop ran `max_it_1b` iterations; the last iterate is returned (Quack's R_qsParquet stops the program there). Raise `max_it_1b`, or turn on one-body DIIS (`max_diis_1b`) | `ParquetNotConverged` | named flag `'parquet_one_body_not_converged'`; `converged_1b` False |
| parquet quasiparticle equation of FLEX, osPA and evPA solved by a function (`qp_solver` a function, `Parquet.options.checked_qp_solver`) | the function's verdict is False for an orbital whose energy is finite; the energy and Z are kept as it returned them | `ParquetNotConverged` | named flag `'parquet_qp_not_converged'`, one naming every such orbital; `qp_converged` False for the orbital |
| parquet quasiparticle equation of FLEX, osPA and evPA, every `qp_solver` (`'find_qp_root'` through the root finder; `'linearized'`; a function, as above) | the energy kept has pole strength Z outside (0, 1], which no root of a positive-weight self-energy has: the finder's root, the linearized solution, or an energy a function returned; it is kept, as Quack keeps it. Such energies sit beside the poles of the regularized self-energy (seen at s_1b 50 and above; the linearized evPA of ammonia/6-31G at s_1b 1e8 has Z = 7862 at its second iteration): check the orbital before using its energy. With spin pairs (an RHF through `run_parquet_so`) both spin orbitals of a pair are named | `QPWeightOutside` | named flag `'z_outside'`, one per orbital and one-body iteration |
| parquet quasiparticle equation of evPA by the root finder (`qp_solver='find_qp_root'`, the drivers' default), a root followed from the previous one-body iteration | the house finder reports that the largest pole strength has left the root followed: its branch has gone (the root of largest weight is kept again) or a root of larger weight lies in the scan window of the root followed (the root followed is kept on its branch). For this self-energy, whose residues have both signs, the finder refines every bracket of that window whatever the Z of the root followed (evPA at s_2b = 1: water/6-31G at the tests' geometry and s_1b = 1e8, third iteration: an orbital kept at Z 0.27 beside a root of Z 0.96 0.02 Ha away; ammonia/6-31G at s_1b = 50: the N 1s kept at Z 0.58 beside a root of Z 0.70) | `RuntimeWarning` | named flag `'parquet_qp_branch_flipped'`, one per orbital and one-body iteration; per orbital the history's `'qp_followed'` and `'qp_flipped'`, the result's for the last iteration |
| parquet front doors (`run_parquet`, `run_parquet_so`) | the mean-field reference is not converged; the parquet run starts from its last iterate. Converge the SCF first | `ParquetNotConverged` | named flag `'scf_not_converged'` |
| parquet two-body loop (as above) | every reducible-vertex change of an iteration is NaN (gfortran's maxval skips NaN unless all are, so Quack ends the loop and reports success); the spectra and vertices returned are NaN. A diverging loop (no TDA, s_2b of 10 to 50 without DIIS, or `run_parquet_so`'s undamped default): turn on two-body DIIS, damp, or lower `eta_2b` | `ParquetNotFinite` | named flag `'parquet_two_body_not_finite'`; `converged_2b` False |
| parquet one-body loop of evPA and qsPA (as above) | the one-body change (max \|eQP - eOld\|, or the qsPA commutator) is NaN, which ends Quack's loop as a success; the energies are NaN | `ParquetNotFinite` | named flag `'parquet_one_body_not_finite'`; `converged_1b` False |
| parquet DIIS (`Parquet.diis.QuackDIIS`: the two-body loop's, and the evPA and qsPA one-body ones) | the Gram matrix of the error history holds NaN or inf (only the errors are tested), where Quack's weights are undefined (dsysvx stops at the NaN pivot and linear_solve ignores its info, so Quack sums uninitialized weights): the extrapolated iterate is NaN in every element. The cause is upstream, a loop that went non-finite. An ill-conditioned system (rcond below eps, routine in two-body DIIS) is solved without a warning, as in Quack; its rcond is in each iteration record of `result.history` (`'diis_rcond'`) | `ParquetNotFinite` | named flag `'parquet_diis_not_finite'` |
| parquet DIIS (as above) | the bordered DIIS system is exactly singular (dsysvx finds a zero pivot, e.g. two equal errors of a loop that stopped moving), where Quack's weights are undefined: its least-squares weights are taken instead, so that step is not Quack's | `ParquetNotFinite` | named flag `'parquet_diis_singular'` |
| qsPA one-body step (`qs_parquet`, `qs_parquet_so`) | the effective Fock matrix holds NaN or inf, or LAPACK failed on it; the energies and orbitals are NaN from there on | `ParquetNotFinite` | named flag `'parquet_qs_fock_not_finite'` |
| parquet BSE solvers (`restricted_bse.solve_eh_bse`, `solve_pp_bse`; `spin_orbital_two_body.solve_eh_bse_so`, `solve_pp_bse_so`) | NaN or inf entered a BSE diagonalization, which Quack lets pass (its LAPACK's output is then arbitrary): the eigenpairs are NaN | `ParquetNotFinite` | named flag `'parquet_bse_not_finite'` |
| parquet diagonal self-energy (`restricted_self_energy.DiagonalSelfEnergy`, `diagonal_self_energy`; `spin_orbital_self_energy.DiagonalSelfEnergySO`, `diagonal_self_energy_so`) | Sigma_pp(w) or its derivative is NaN or inf: an energy, a spectrum or an integral that is not finite, an overflow, or (the closed-shell diagonal) dSigma/dw exactly 1, Z = 1/0; returned as computed. A denominator exactly zero (w on a pole, a zero mode) is no event: where Quack's formula gives 0/0 its term is 0 in Sigma, the limit, and 0 in dSigma/dw, the value of Quack's slope a few ulps off zero (its limit would be -2 s_1b times the residue), so the term adds nothing to Sigma or Z whether rounding puts a zero mode at 0 or a few ulps off | `ParquetNotFinite` | named flag `'parquet_self_energy_not_finite'` |
| qsPA static self-energy (`restricted_static_self_energy.static_self_energy_srg`, `static_self_energy_eta`; `spin_orbital_self_energy.static_self_energy_srg_so`) | an element of the static matrix is NaN or inf (an energy, a spectrum or an integral that is not finite, or an overflow), as in Quack; returned as computed. A denominator exactly zero is no event: the SRG forms take the limit 0 (of R(d) and of the kernel at D_p = D_q = 0) where Quack's gives 0/0, and the eta form's x / (x^2 + eta^2) is 0 there at every broadening whose square does not underflow, above about 1.6e-162 Ha (below it, and at 0, that term is 0/0, NaN and flagged, as in Quack; the driver takes any broadening > 0) | `ParquetNotFinite` | named flag `'parquet_static_sigma_not_finite'` |
| parquet quasiparticle equation solved by a function (as above) | the function returned an energy that is not finite; it is kept as returned | `ParquetNotFinite` | named flag `'parquet_qp_not_finite'`, one naming every such orbital |
| parquet reducible vertices (`restricted_vertices.regularized_inverse` under `eh_reducible_vertex`, `pp_reducible_vertex`; `spin_orbital_two_body.eh_reducible_vertex_so`, `pp_reducible_vertex_so`) | an energy is NaN (or infinite at s_2b = 0), so a vertex weight is not finite: the vertex is NaN there, as in Quack (both branches count the weights that are not finite). A denominator omega -+ Omega exactly zero is no event: its weight is the limit 0, where Quack's regularized 1/d is 0/0. H/STO-3G's spin flip, e_beta - e_alpha - J, zero in exact arithmetic, comes out 2.2e-16 or exactly 0 as the mean-field energies and J round, and gives the same run either way in the eh TDA; without it the full eh solution's X + Y = Omega^-1/2 Z^T (A - B)^1/2 of an exactly zero root is 0 times inf and the run NaN, as in Quack | `ParquetNotFinite` | named flag `'parquet_vertex_pole'` |
| parquet vertex hook (the drivers' `vertex_builder`, any builder but the default `static_reducible_vertex`, `static_reducible_vertex_so`) | the builder returned a vertex holding NaN or inf (counted before the damping); it is kept as returned, and since the change max \|Phi_new - Phi_old\| passes over NaN (gfortran's maxval) the loop can report convergence with it. An output that is no real array of the vertex's shape is refused (ValueError) | `ParquetNotFinite` | named flag `'parquet_vertex_not_finite'` |
| parquet eh BSE without the TDA (`solve_eh_bse`, `solve_eh_bse_so`) | A - B is not positive definite: (A - B)^1/2 is NaN, and with it the whole eh solution, as in Quack. The kernel made the problem unstable: the TDA (`tda_eh=True`) or a smaller `eta_2b` | `ParquetBSEUnstable` | named flag `'parquet_eh_a_minus_b_not_positive'` |
| parquet eh BSE without the TDA (as above) | negative squared excitation energies: the closed-shell solver takes sqrt\|Omega^2\| (Quack's phRLR), the spin-orbital one gives NaN (Quack's phGLR) | `ParquetBSEUnstable` | named flag `'parquet_eh_negative_excitations'` |
| parquet eh BSE in the TDA (`solve_eh_bse`, `solve_eh_bse_so`) | a negative excitation energy, which Quack lets pass; kept. A closed shell past its triplet instability (H2/6-31G at 2.0 Angstrom: -0.063 Ha in both drivers), vertices that diverged, or an open shell in evPA and qsPA: from the second one-body iteration on the quasiparticle energies narrow the gap of a spin flip below its exchange term, so the eh TDA problem has negative spin-flip roots before any kernel (OH UHF/6-31G after one iteration: -0.18 Ha in evPA, -0.23 Ha in qsPA) and every later two-body iteration is flagged; at s_2b = 1 both still converge, and osPA, on the Hartree-Fock energies, raises no such flag. Over spin orbitals only a root below -`PARQUET_EH_ZERO_MODE_TOL` (-1e-3 Ha) counts: a one-electron system's spin flip is an exact zero that roundoff puts on either side | `ParquetBSEUnstable` | named flag `'parquet_eh_tda_negative_excitations'` |
| parquet eh BSE without the TDA (`solve_eh_bse`, `solve_eh_bse_so`) | an excitation energy below `PARQUET_EH_ZERO_MODE_TOL` (1e-3 Ha), which Quack lets pass: over spin orbitals the spin rotation of a collinear open shell, an exact zero mode that the SCF residual moves to 1e-7 to 1e-4 Ha; in either driver a closed shell at the edge of an instability (H2/6-31G 1e-6 Angstrom before its triplet instability: 3.6e-4 Ha). Its X + Y ~ Om^-1/2 takes over the vertices and the self-energy (which grows as 1/Om, Z going to zero), so the energies are finite and meaningless (NH2 UHF/6-31G FLEX: up to 18 eV from the TDA; that H2 at s_1b = 50: every quasiparticle root of Z below 0.012, the HOMO's at +22.9 eV); kept. Run open shells in the eh TDA (`tda_eh=True`, the default), at a moderate `eta_2b`: in the TDA these modes are small rather than zero and not flagged (OH UHF/6-31G: 7.5e-4 Ha for the rotation of its pi hole, 3.0e-3 and 7.4e-3 Ha for spin flips), and a vertex weighs such a root by about 1/Om once 2 s_2b Om^2 nears 1 (OH's osPA, two-body DIIS 5 and damping 0.75: converged at s_2b = 1, diverged at 1e6 and 1e8) | `ParquetBSEUnstable` | named flag `'parquet_eh_zero_mode'`, at each two-body iteration where the root stays low |
| parquet pp BSE without the TDA (`solve_pp_bse`, `solve_pp_bse_so`) | other than nVV eigenvalues above zero: as Quack's sort_ppRPA, the first ones in dgeev's order (which differs between LAPACK libraries) are kept and missing ones padded with zeros, which makes that branch inf/NaN | `ParquetBSEUnstable` | named flag `'parquet_pp_branch_count'` |
| parquet pp BSE without the TDA, spin orbitals (`solve_pp_bse_so`) | complex eigenvalues beyond roundoff (`PARQUET_PP_COMPLEX_TOL`): their real parts, and the real and imaginary parts of their eigenvectors, are used as roots, as in Quack | `ParquetBSEUnstable` | named flag `'parquet_pp_complex_roots'` |
| parquet pp BSE without the TDA (`solve_pp_bse`, `solve_pp_bse_so`) | the metric overlap of a branch has an eigenvalue below `PARQUET_LOWDIN_SMALL_EIGENVALUE` (Quack's orthogonalize_matrix warning); its inverse square root is taken regardless, NaN for a negative one | `ParquetBSEUnstable` | named flag `'parquet_pp_lowdin_small_eigenvalue'` |
| parquet pp BSE (`solve_pp_bse`, `solve_pp_bse_so`) | EcRPA and the correlation energies of the 2p and 2h branches differ by more than `PARQUET_PP_EC_CONSISTENCY_TOL` (Quack's 'RPA1 != RPA2'); kept | `ParquetBSEUnstable` | named flag `'parquet_pp_ec_inconsistent'` |
| parquet pp BSE without the TDA (`solve_pp_bse`, `solve_pp_bse_so`) | dgeev did not converge; the eigenpairs are what it returned, as in Quack | `ParquetBSEUnstable` | named flag `'parquet_pp_dgeev_failed'` |
| parquet BSE solvers (as above) | a symmetric BSE diagonalization failed in LAPACK: its eigenpairs are NaN | `ParquetBSEUnstable` | named flag `'parquet_bse_diagonalization_failed'` |

The parquet drivers (`SingleReference.Parquet`, ported from Quack) warn with
three classes of their own, held in `Parquet.events` beside the names of
their flags; each covers both the closed-shell and the spin-orbital branch,
so one filter sees an event from either:

- `ParquetNotConverged`: a parquet loop or search stopped short, or its
  reference was not converged. The number returned is the last iterate.
- `ParquetNotFinite`: a quantity came out NaN or infinite, or the arithmetic
  met an exact zero where Quack's result is undefined (a DIIS pivot). The
  values are returned as computed; where Quack's are NaN, so are these, but
  for an exactly zero denominator of a regularized weight, where the weight
  and its slope are 0 (no event).
- `ParquetBSEUnstable`: an eh or pp Bethe-Salpeter problem of the two-body
  loop is unstable or its diagonalization failed, Quack's printed BSE
  warnings and two events it lets pass (a negative Tamm-Dancoff excitation,
  an eh zero mode). The spectrum is what Quack's arithmetic gives.

The photophysics stages gather every named flag raised inside them, and
their own events, into the stage record's `numerical_flags`, and the rate
table gathers those by stage, so it says which numbers stand on a flagged
stage.

G0W@CC (`calc_qp_energy(..., polarizability='CCSDT', out=out)`) records the
CC and EOM events above in `out['numerical_flags']`: a CC amplitude solve out
of iterations, unconverged EOM roots, and an unpaired or complex EOM state.

A matrix-free BSE solve can be made strict again:
`solve_bse_isdf(..., refuse_unconverged=True)` raises `DavidsonNotConverged`
instead of flagging.

## Example

An evGW loop cut off after two cycles returns its last iterate, warns, and
says so in its record:

```python
import warnings

from pyscf import dft, gto

from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.evGW import evgw_eigenvalues

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
mf = dft.RKS(mol, xc='pbe0').density_fit().run()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    eps, info = evgw_eigenvalues(mf, mol, mode='casida',
                                 numerics=Numerics(evgw={'max_cycle': 2}))

print('converged:', info['converged'])        # False
for w in caught:
    print(type(w.message).__name__, w.message)
```
