# Physical fidelity and Twin refinement protocol

This protocol is the dev12 software-side handoff from a completed Physical run to a
justified Twin correction. It does **not** create Physical evidence and does not authorize
model changes from bundled smoke data.

## 1. Preserve matched Real and Twin runs

Run the Physical Study and the corresponding Twin configuration with matched workload,
seed, application, topology declaration, and starting model state. Keep each canonical
`events.jsonl` in its sealed run/Study artifact.

Create a fidelity-batch manifest with stable pair IDs:

```yaml
schema: darpan.fidelity-batch/v1
name: physical-reference
refinement_policy: refinement-policy.yaml
require_sealed_inputs: true
pairs:
  - id: seed-201
    real: physical/seed-201/events.jsonl
    twin: twin/seed-201/events.jsonl
```

The pair ID is evidence identity, not a generated runtime instance ID. For paper-grade
evidence, point `real` and `twin` at sealed experiment-artifact directories (or traces
covered by a parent artifact manifest) and keep `require_sealed_inputs: true`. Darpan
verifies each source artifact first and carries its manifest fingerprint into the batch.

## 2. Freeze the refinement policy before inspecting residuals

A paper-grade batch should bind `refinement_policy` in the batch manifest. The policy
contains explicit sample sufficiency, acceptable MAE, normalized-error contribution, and
target rules. MAE thresholds remain metric-native (for example seconds or bytes).
Example shape:

```yaml
schema: darpan.twin-refinement-policy/v1
minimum_total_samples: 30
maximum_uncovered_error_contribution: 0.20
max_candidates: 2
rules:
  - metric: execution_duration
    target: twin.models.execution
    minimum_samples: 10
    maximum_acceptable_mae: 0.20
    minimum_error_contribution: 0.15
    required: true
```

The numerical values above are illustrative. Final thresholds must be chosen from the
research protocol, not copied from this document after viewing the Physical results.

## 3. Build sealed multi-run fidelity evidence

```bash
darpan fidelity-batch fidelity-pairs.yaml --output evidence/fidelity-batch
darpan artifact verify evidence/fidelity-batch
```

The artifact contains:

- the original batch manifest;
- the predeclared policy when configured;
- copied Real/Twin canonical traces for every pair;
- source artifact manifests/fingerprints when inputs are sealed;
- per-pair fidelity reports;
- aggregate fidelity summaries;
- `fidelity-diagnosis.json`;
- `residuals.csv`;
- a recursive `artifact-manifest.json`.

Each metric keeps MAE/RMSE in its native unit (seconds for time metrics, bytes for
artifact size). Cross-metric contribution and ranking use dimensionless normalized error,
so a byte count cannot dominate a time residual merely because of its unit scale. Entity
and matched-pair rankings use the same normalized basis; pair ranking helps identify an
anomalous run before changing the Twin model.

## 4. Evaluate the refinement gate

```bash
darpan refinement evidence/fidelity-batch --output evidence/refinement-decision
darpan artifact verify evidence/refinement-decision
```

The gate returns:

- `insufficient_evidence`: total or required-metric sample counts are too small;
- `manual_review`: a material residual is outside the predeclared metric policy;
- `refine`: one or more declared thresholds are exceeded, with only the corresponding
  `authorized_targets` permitted by this evidence;
- `hold`: evidence is sufficient and no declared refinement threshold is exceeded.

The gate never modifies code, parameters, or model snapshots. It only creates a sealed,
auditable decision artifact.

If the batch sealed a policy, providing a different `--policy` later is rejected. A policy
passed only after evidence exists is supported for exploration, but the output records
`policy_predeclared_with_evidence: false` and should not be presented as preregistered
paper methodology.

## 5. Change only evidence-authorized scope

A `refine` decision is a permission boundary, not proof that a particular implementation
change is correct. Inspect the dominant entities and Physical traces, implement only the
smallest justified model change, then rerun the same held-out validation protocol.

Do not add packet loss, jitter, memory bandwidth, I/O, GPU, energy, thermal, or other Twin
complexity unless Physical evidence and the frozen protocol justify that layer.

## 6. Freeze after correction

Once the reference Physical data has justified and validated any correction, freeze the
Twin implementation/model protocol before the final six-capability and Real-only versus
Twin-assisted learning studies. Final-study results must not become another model-tuning
set.
