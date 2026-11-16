"""Pulay's DIIS as Quack's parquet loops use it (utils/DIIS_extrapolation.f90).

The history holds the last `size` pairs (error, iterate), newest first; each
step adds one pair and returns the extrapolated iterate

    x = sum_k w_k x_k,   [[E^T E, -1], [-1, 0]] (w; lambda) = (0; -1),

which is not itself stored. A history of one returns the iterate unchanged,
so `size` 1 is no DIIS at all, as Quack's max_diis 1 is (Quack's drivers then
never call DIIS_extrapolation, and the drivers here keep no history).

ONE VECTOR IN BLOCKS. A vector of the history is a tuple of blocks of fixed
shapes, and its dot product the sum of the blocks' ones: the four closed-shell
reducible vertices, each (norb,)*4 (Quack's one vector of 4 norb^4 numbers,
eh singlet, eh triplet, pp singlet, pp triplet), the two spin-orbital ones
(eh, pp), or one block, the quasiparticle energies of evPA or the Fock matrix
of qsPA. With `weights` the dot product weighs each block's by its own
number: the per-insertion two-body loops weigh each tensor of a vertex by
one over the number of its tensors (an eh vertex's two roles a half each),
so that each vertex counts once, as in Quack's vector.

THE STEP. The errors and iterates live in `size` slots allocated once. The
caller takes the next slot (`next_slot`, once the history is full the oldest
pair's), writes the new pair into its arrays in place, and calls
`extrapolate`, which

* adds one row to the Gram matrix E^T E, the new error's dot products with
  the errors kept (Quack recomputes the whole matrix with one dgemm; its upper
  triangle, which dsysvx reads, pairs each error with the older ones as this
  row does), and hands the matrix to the solver newest first, the order
  Quack's prepend gives its history;
* solves the bordered system with LAPACK dsysvx ('N', 'U': Bunch-Kaufman, a
  condition estimate and iterative refinement), as Quack's linear_solve does,
  and keeps its reciprocal condition number (`rcond`, `rcond_history`; the
  drivers record it in every iteration record as 'diis_rcond'). A system
  below rcond = eps, routine in two-body DIIS (cond 1e14 to 1e18), is solved
  all the same and without a warning, as in Quack: rounding sets its
  weights, and hardly the extrapolated iterate, since the errors it
  combines are nearly one vector (two-body DIIS runs agree with Quack's to
  about 1e-12 Ha where each loop starts its history afresh, to about
  1e-11 Ha with one-body DIIS as well, to about 1e-9 Ha where the history
  carries over between evPA iterations: the package docstring);
* accumulates sum_k w_k x_k, newest first, into the arrays the caller hands
  in (`out`), through one scratch block of at most PARQUET_DIIS_BLOCK_BYTES
  and 1/PARQUET_MIN_BLOCKS_PER_VERTEX of the largest block, so no temporary
  of the size of one vertex exists.

`push(error, iterate)` does the three in one call for the small one-body
vectors and returns new arrays.

MEMORY. The slots hold 2 size vectors and nothing else is kept between
steps: for the closed-shell vertices 8 size norb^4 numbers (116 GB at
norb = 138 and size 5; 12 size norb^4 for the six tensors of a
per-insertion loop). A step adds its scratch block and nothing more, so
the extrapolation stays below the vertex builds of the same iteration, and
the two-body loop peaks at its peak without DIIS plus 8 size norb^4 numbers:
on water/aug-cc-pVDZ (norb 41) 7.37 norb^4 without DIIS and 47.37 with
size 5, of which 7 and 47 are the arrays the loop holds, the integrals
<pq|rs> among them, and the rest the vertex builds' temporaries
(`restricted`, COST AND MEMORY). Quack's R_evParquet holds 17 + 8 size
norb^4 numbers besides the integrals it is handed (its own memory count
leaves them out), 18 + 8 size counted as here.

Two events keep their numbers and are flagged (`events`):

* a NaN or inf in the Gram matrix of the history (an error, or the iterate it
  was formed from, went non-finite; only the errors are tested): the iterate
  returned is NaN in every element ('parquet_diis_not_finite'). Quack's
  weights are undefined there: dsysvx stops at the NaN pivot without a
  solution and linear_solve ignores its info, so Quack's DIIS_extrapolation
  sums uninitialized weights. A non-finite iterate beside finite errors enters
  the sum as it is, as in Quack;
* an exactly singular system (dsysvx finds a zero pivot), where Quack's
  weights are undefined: the minimum-norm least-squares weights are taken
  instead ('parquet_diis_singular').
"""
import numpy as np
import scipy.linalg as la
from scipy.linalg import lapack

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import PARQUET_DIIS_BLOCK_BYTES, PARQUET_MIN_BLOCKS_PER_VERTEX
from src.SingleReference.Parquet.events import (FLAG_DIIS_NOT_FINITE, FLAG_DIIS_SINGULAR,
                                                ParquetNotFinite)


class QuackDIIS:
    """A DIIS history of at most `size` (error, iterate) pairs over vectors in
    blocks of the given shapes (module docstring).

        diis = QuackDIIS(5, [(norb,) * 4] * 4, label='two-body')
        errors, iterates = diis.next_slot()     # fill each block in place
        diis.extrapolate(out=vertices)          # vertices <- sum_k w_k x_k

    Parameters
    ----------
    size : int
        Largest number of pairs kept (Quack's max_diis_1b, max_diis_2b); 1 is
        no DIIS, the iterate coming back unchanged.
    shapes : sequence of tuple of int
        The shape of each block of one vector.
    label : str
        Names the history in the flags' messages ('two-body', 'evPA one-body').
    weights : sequence of float, optional
        One positive finite number per block, the weight of its dot product
        in the Gram matrix; None (the default, Quack's) weighs every block 1
        and sums the plain dot products.

    Attributes
    ----------
    rcond : float or None
        dsysvx's reciprocal condition number of the last system (NaN where
        the history was not finite; None before the first system).
    rcond_history : list of float
        The rcond of every system solved since the history was made.
    """

    def __init__(self, size, shapes, label='DIIS', weights=None):
        if int(size) != size or size < 1:
            raise ValueError(f'a DIIS history holds at least one pair, not {size!r}')
        self.size = int(size)
        self.shapes = tuple(tuple(int(n) for n in shape) for shape in shapes)
        if not self.shapes:
            raise ValueError('a DIIS vector has at least one block')
        self.weights = None if weights is None else tuple(float(w) for w in weights)
        if self.weights is not None and (
                len(self.weights) != len(self.shapes)
                or not all(0.0 < w < np.inf for w in self.weights)):
            raise ValueError(f'DIIS block weights {weights!r}: one positive finite number per '
                             f'block, {len(self.shapes)} blocks')
        self.label = str(label)
        self._errors = [tuple(np.empty(s) for s in self.shapes) for _ in range(self.size)]
        self._iterates = [tuple(np.empty(s) for s in self.shapes) for _ in range(self.size)]
        self._gram = np.zeros((self.size, self.size))
        self.rcond = None
        self.rcond_history = []
        self.reset()

    def __len__(self):
        return len(self._order)

    def reset(self):
        """Forget the history and keep the slots (R_qsParquet resets its
        two-body history at every one-body iteration)."""
        self._order = []                    # the slots in the history, newest first
        self._free = list(range(self.size))
        self._pending = None                # the slot next_slot handed out

    def next_slot(self):
        """(errors, iterates): the blocks of the slot the next pair goes into,
        each an array of its block's shape to be written in place before
        `extrapolate`. Once the history is full it is the oldest pair's slot:
        that pair leaves the history here, so the history never holds a slot
        being written. Asked again before `extrapolate`, the same slot."""
        if self._pending is None:
            self._pending = self._free.pop(0) if self._free else self._order.pop()
        return self._errors[self._pending], self._iterates[self._pending]

    def extrapolate(self, out=None):
        """Add the pair written into `next_slot` to the history and return the
        extrapolated iterate.

        Parameters
        ----------
        out : sequence of ndarray, optional
            C-contiguous float64 arrays of the block shapes that receive the
            extrapolated blocks (they may be the arrays the new iterate was
            damped from, not arrays of this history); new arrays by default.

        Returns
        -------
        tuple of ndarray
            The blocks of the extrapolated iterate (the arrays of `out` when
            given).
        """
        if self._pending is None:
            raise RuntimeError('extrapolate adds the pair written into next_slot(): '
                               'take the slot first')
        out = self._outputs(out)
        s, self._pending = self._pending, None
        self._order.insert(0, s)
        if self.size == 1:
            for o, x in zip(out, self._iterates[s]):
                np.copyto(o, x)
            return out
        new = self._errors[s]
        for j in self._order:
            g = 0.0
            if self.weights is None:
                for a, b in zip(new, self._errors[j]):
                    g += float(np.dot(a.reshape(-1), b.reshape(-1)))
            else:
                for weight, a, b in zip(self.weights, new, self._errors[j]):
                    g += weight * float(np.dot(a.reshape(-1), b.reshape(-1)))
            self._gram[s, j] = self._gram[j, s] = g
        n = len(self._order)
        a = np.zeros((n + 1, n + 1))
        a[:n, :n] = self._gram[np.ix_(self._order, self._order)]
        a[:n, n] = a[n, :n] = -1.0
        b = np.zeros((n + 1, 1))
        b[n, 0] = -1.0
        w = self._weights(a, b)
        if w is None:
            for o in out:
                o.fill(np.nan)
        else:
            self._accumulate(w[:n], out)
        return out

    def push(self, error, iterate):
        """Write (error, iterate) into the next slot and return the
        extrapolated iterate as new arrays: the one-call form for the small
        one-body vectors. error and iterate are arrays of the block shapes
        (one array for a one-block history, else a sequence of them); the
        result likewise. Wrong blocks are refused (ValueError) before the
        history changes."""
        if len(self.shapes) == 1:
            error, iterate = (error,), (iterate,)
        sources = [np.asarray(src, dtype=float) for src in tuple(error) + tuple(iterate)]
        if len(sources) != 2 * len(self.shapes):
            raise ValueError(f'{self.label} DIIS: {len(sources)} blocks for an error and an '
                             f'iterate of {len(self.shapes)} blocks each')
        for src, shape in zip(sources, self.shapes * 2):
            if src.shape != shape:
                raise ValueError(f'{self.label} DIIS: a block of shape {src.shape}, '
                                 f'expected {shape}')
        errors, iterates = self.next_slot()
        for dst, src in zip(errors + iterates, sources):
            np.copyto(dst, src)
        result = self.extrapolate()
        return result[0] if len(self.shapes) == 1 else result

    def _outputs(self, out):
        """The arrays the extrapolation is written into: `out`, checked, or new."""
        if out is None:
            return tuple(np.empty(s) for s in self.shapes)
        out = tuple(out)
        if len(out) != len(self.shapes):
            raise ValueError(f'{self.label} DIIS: {len(out)} output blocks for '
                             f'{len(self.shapes)}')
        slots = [x for pair in (self._errors, self._iterates) for slot in pair for x in slot]
        for o, s in zip(out, self.shapes):
            if not (isinstance(o, np.ndarray) and o.shape == s and o.dtype == np.float64
                    and o.flags.c_contiguous and o.flags.writeable):
                raise ValueError(f'{self.label} DIIS: an output block must be a writable '
                                 f'C-contiguous float64 array of shape {s}')
            if any(np.may_share_memory(o, x) for x in slots):
                raise ValueError(f'{self.label} DIIS: an output block lies in the history')
        return out

    def _weights(self, a, b):
        """The weights w of the bordered system a (w; lambda) = b: dsysvx
        ('N', 'U'); the least-squares w when it is exactly singular (flagged);
        None when a is not finite (flagged)."""
        n = a.shape[0]
        if not np.isfinite(a).all():
            self.rcond = float('nan')
            self.rcond_history.append(self.rcond)
            flag(FLAG_DIIS_NOT_FINITE,
                 f'{self.label} DIIS: the Gram matrix of its error history holds NaN or '
                 f'inf (an error or an iterate went non-finite), where Quack\'s weights '
                 f'are undefined (dsysvx stops at the NaN pivot and linear_solve ignores '
                 f'its info); the extrapolated iterate is NaN in every element',
                 ParquetNotFinite, stacklevel=3)
            return None
        # Quack's linear_solve: the optimal workspace from a query, else LAPACK's least
        lwork, info = lapack.dsysvx_lwork(n)
        work = {'lwork': int(lwork)} if info == 0 else {}
        _, _, _, _, x, rcond, _, _, info = lapack.dsysvx(a, b, **work)
        if info < 0:
            raise RuntimeError(f'dsysvx: illegal value of argument {-info}')
        self.rcond = float(rcond)
        self.rcond_history.append(self.rcond)
        if 0 < info <= n:
            flag(FLAG_DIIS_SINGULAR,
                 f'{self.label} DIIS: the system of its {n - 1} pairs is exactly singular '
                 f'(dsysvx finds a zero pivot, info {info}), where Quack\'s weights are '
                 f'undefined; the extrapolation takes its least-squares weights',
                 ParquetNotFinite, stacklevel=3)
            return la.lstsq(a, b[:, 0])[0]
        return x[:, 0]

    def _accumulate(self, w, out):
        """out <- sum_k w_k x_k over the history, newest first, block by block:
        for every element, ((0 + w_1 x_1) + w_2 x_2) + ..., the products
        rounded before they are added. One scratch block serves every block
        of the vector: at most PARQUET_DIIS_BLOCK_BYTES and
        1/PARQUET_MIN_BLOCKS_PER_VERTEX of the largest block (the sum of each
        element does not depend on the block)."""
        largest = max(o.nbytes for o in out)
        step = max(1, min(PARQUET_DIIS_BLOCK_BYTES, largest // PARQUET_MIN_BLOCKS_PER_VERTEX)
                   // out[0].itemsize)
        scratch = np.empty(min(step, max(o.size for o in out)))
        for k, o in enumerate(out):
            target = o.reshape(-1)
            sources = [self._iterates[j][k].reshape(-1) for j in self._order]
            for start in range(0, target.size, step):
                acc = target[start:start + step]
                t = scratch[:acc.size]
                acc.fill(0.0)
                for wk, xk in zip(w, sources):
                    np.multiply(xk[start:start + step], wk, out=t)
                    acc += t
