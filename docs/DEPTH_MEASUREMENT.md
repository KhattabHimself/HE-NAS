# Measuring CKKS Level Consumption

Supporting document for §III of the HE-NAS paper. Everything here is a
measurement taken on this project's own encryption path. Nothing is derived
analytically, and nothing depends on any prior published model.

---

## 1. Why this had to be measured

Two quantities are easily conflated.

| Quantity | Definition | Determined by |
|---|---|---|
| **Multiplicative depth** | Longest chain of dependent ciphertext×ciphertext products | The arithmetic circuit |
| **Level consumption** | Number of primes the library removes from the modulus chain | How often the library *rescales* |

A search constraint has to respect the second. Rescaling is triggered by scale
growth, and scale growth is caused by *any* multiplication that raises the
scale, including ciphertext×plaintext products that add no circuit depth.

The two coincide only under assumptions about scale management and polynomial
evaluation. We did not want an evolutionary feasibility oracle resting on
assumptions, so we measured.

**Cost of measuring:** ~1 minute (per-operation) + ~10 minutes (whole-network).
**Cost of the search that depends on it:** ~950 hours.

---

## 2. Method

Identical protocol throughout, and it is a direct measurement rather than an
inference:

1. Build a well-formed chain `[60, 40×B, 60]`, giving `B` usable levels.
2. Set `global_scale = 2^40`.
3. Run the operation (or full forward pass) on a freshly encrypted vector.
4. Decrypt. Accept if it returns and all outputs are finite within tolerance.
5. Increment `B` from 0 and report the smallest `B` that succeeds.

Failure below the threshold is not graceful. TenSEAL raises
`ValueError: scale out of bounds`, or returns values that are not finite. There
is no accuracy taper to interpolate through, which is what makes the smallest
successful `B` an exact measurement rather than an estimate.

---

## 3. Per-operation costs

Each primitive isolated, run alone on a freshly encrypted vector.

`python probe_decomposition.py` · `python probe_ops.py`

| Operation | Levels consumed |
|---|---|
| Encryption only (baseline) | **0** |
| Ciphertext × plaintext vector (`mul`) | **1** |
| Ciphertext × plaintext matrix (`mm`) | **1** |
| `square()` | **1** |
| `polyval`, degree 2 | **1** |
| `polyval`, degree 4 | **3** |
| `mul` then `polyval` degree 2 | **2** |
| `mul` then `polyval` degree 4 | **4** |
| `mm` then `square` | **2** |

Two mechanisms account for this. **Neither is novel.** Both are documented
behaviour of CKKS and of the library, and are stated here because the search
constraint depends on them, not because they are new (see §7).

**(a) A ciphertext×plaintext product consumes one level.** It contributes no
ciphertext×ciphertext multiplication, but it raises the scale from Δ to Δ²
exactly as a ciphertext product would, and the library rescales to restore it.
The level is spent on scale management, not on depth. This holds for both
element-wise (`mul`) and matrix (`mm`) forms, so convolutions and linear layers
cost the same.

**(b) `polyval` consumes `d−1` levels for degree `d`.** A degree-`d` polynomial
admits evaluation in `⌈log₂ d⌉` sequential multiplications by
Paterson–Stockmeyer (1973), and that bound is achievable, but attaining it
requires a baby-step/giant-step evaluator. TenSEAL evaluates sequentially.
Verified by inspection as well as measurement: `CKKSVector` exposes only
`polyval` and `pow`, and grepping the installed package for
`paterson|stockmeyer|baby.step|giant.step|bsgs` returns nothing.

**Costs are additive under composition** — row 7 = row 2 + row 5, row 8 = row 2
+ row 6, row 9 = row 3 + row 4. This is what licenses summing per-layer costs.

---

## 4. The rule

A layer costs 1 level for its convolution plus `d−1` for its degree-`d`
activation, so per layer the cost is exactly `d`:

```
levels = Σ_i d_i           (sum of activation polynomial degrees)
chain length |q| = levels + 2
```

The two trailing primes maintain the base level and provide output-scale
headroom.

| Activation | Degree | Levels per layer |
|---|---|---|
| `square` (x²) | 2 | **2** |
| `relu4` | 4 | **4** |

**Corollaries.** A depth-`d` network of `square` activations needs `2d` levels,
i.e. **`2d+2` primes**. An MLP of `d` hidden layers, having `d+1` linear layers
and `d` squares, needs `2d+1` levels, i.e. **`2d+3` primes**.

Note that the per-layer cost is *linear* in the activation degree, not
logarithmic. At higher degrees the activation term dominates the budget: degree
8 would cost 8 levels per layer, not 3.

---

## 5. Whole-network validation

`python validate_depth_model.py` (~10 min, CPU only)

Full encrypted forward passes through the project's own validator, unmodified,
so this exercises the same encryption path that later validates evolved
champions.

| Architecture | Measured | Rule | Match | Latency |
|---|---|---|---|---|
| 1× square | 2 | 2 | ✓ | 7 ms |
| 1× relu4 | 4 | 4 | ✓ | 35 ms |
| 2× square | 4 | 4 | ✓ | 42 ms |
| 3× square | 6 | 6 | ✓ | 89 ms |
| relu4, square | 6 | 6 | ✓ | 85 ms |
| square, 2× relu4 | 10 | 10 | ✓ | 438 ms |

**6/6.** Relative error at the accepted budget is ≤ 6.3e-5 throughout, so these
are correct decryptions, not marginal ones.

### Scaling check

Confirming linear growth rather than a fitted constant, on `mm`+`square` layers:

| Hidden layers `d` | Measured levels | `2d+1` |
|---|---|---|
| 1 | 3 | 3 |
| 2 | 5 | 5 |
| 3 | 7 | 7 |
| 4 | 9 | 9 |
| 6 | 13 | 13 |
| 8 | 17 | 17 |

---

## 6. Cross-version reproduction

The full procedure was run on two independent installations:

| | Laptop | Cluster node |
|---|---|---|
| TenSEAL | 0.3.16 | 0.3.17 |
| NumPy | 1.26 | 2.4 |
| OS | Windows 11 | RHEL 9 (Alliance Canada Nibi) |
| CPU | — | Xeon Platinum 8480+ |
| **Level counts** | **identical** | **identical** |

Level consumption is a deterministic property of the circuit and the evaluator,
not of the host. The agreement confirms it.

---

## 7. Scope and limits

Stated plainly, because both halves of the rule are implementation-dependent:

- **Degrees 2 and 4 only.** We claim nothing about other degrees. Degree 8 is
  the obvious extension and is untested.
- **The convolution cost reflects a library that rescales after every plaintext
  multiply.** An implementation encoding plaintext weights at a reduced scale
  can avoid that rescale, at a cost in weight precision. SEAL permits choosing
  the encoding scale; TenSEAL does not expose it.
- **The activation cost reflects a sequential evaluator.** A library exposing
  baby-step/giant-step evaluation would consume `⌈log₂ d⌉` instead.
- **TenSEAL only.** Reproduced across two versions of it, not across two
  independent HE libraries. Confirming in OpenFHE would close that gap.
- **None of this is a novel result.** That a ciphertext×plaintext product forces
  a rescale, and that rescaling drops a prime, is standard CKKS and is stated
  directly in the scheme's own literature and in library documentation. See
  Cheon et al. (2017) §3 on rescaling; Microsoft SEAL issue
  [#445](https://github.com/microsoft/SEAL/issues/445), where the maintainers
  note that plaintext masking "consumes a level"; nGraph-HE2
  ([arXiv:1908.04172](https://arxiv.org/pdf/1908.04172)); and TenSEAL Tutorial 2
  on approximate numbers. The FHE-compiler literature exists substantially to
  manage this budget.

**What this document is, then.** Not a discovery. A record that the feasibility
oracle used by a 950-hour search was measured against the library that would
execute its output, rather than assumed — and a record of what happened in the
one condition where it was not (§8). Anyone adopting a different activation set,
library, or evaluator should re-run the measurement rather than import these
numbers. It takes about eleven minutes.

---

## 8. Consequence

The constraint is binding, not soft. A configuration provisioned below the
measured requirement does not lose accuracy or run slowly — it trains normally,
reports ordinary plaintext accuracy, and then fails to decrypt.

Condition **B4** in the study demonstrates this. It provisions 6 levels where
the rule requires 12, having been budgeted by counting only ciphertext×ciphertext
products at `⌈log₂ d⌉` per activation. Result: plaintext accuracy 0.6022,
statistically ordinary; fitness 0.0300; **0/30 seeds feasible**; champion fails
to decrypt under real CKKS.

Ten of the other eleven champions execute correctly under real encryption using
the chains their own search selected.

---

## 9. Reproducing everything

```bash
python probe_decomposition.py            # ~1 min   per-primitive isolation
python probe_ops.py                      # ~1 min   mul vs mm vs square vs polyval
python validate_depth_model.py     # ~10 min  whole-network validation
python main.py selftest                  # ~1 min   asserts the rule in code
```

`main.py selftest` asserts the rule directly, in both directions: a depth-`d`
square net runs at `2d+2` primes and **fails** at `2d+1`. A one-sided check
would pass under any over-estimate, so the negative half is the one that
constrains.

**Raw data:** `measurements/depth_validation.json` (cluster, TenSEAL 0.3.17),
`measurements/depth_validation.quick_local.json` (laptop, TenSEAL 0.3.16).
**Rule as implemented:** `ckks.py`, `ACTIVATION_LEVEL_COST` and `CONV_LEVEL_COST`.
**Assertions:** `analysis.py`, `run_selftest`.
