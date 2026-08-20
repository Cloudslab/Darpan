# Architecture contracts

These contracts are intentionally small and are the compatibility boundary for
future Darpan releases.

## Canonical domain

- `Event` records what happened. It carries event time, ingest time, source
  sequence, correlation/causation IDs, a schema version, and an extensible
  payload.
- `ContinuumState` is a materialized view derived from the append-only EventLog; node and link telemetry are materialized on their canonical targets.
- `Action` is a proposal. Validation and arbitration occur before a backend
  applies it.
- `Measurement` represents extensible system quantities shared by physical and
  digital runtimes.
- Resources are named, so accelerators and other future capacities do not
  require new fixed fields in Core.

### Stable action surface

The stable controller vocabulary is exactly `PLACE`, `STOP`, `RESTART`,
`MIGRATE`, `SCALE`, and `ROUTE`. Vocabulary entries are implementation claims
only when the active backend advertises the corresponding capability. Twin
supports all six. Real supports `ROUTE` only with a configured physical
`NetworkControlDriver`; it must otherwise reject the action before dispatch.

`SCALE` controls runtime replicas of one logical long-running component. It does
not clone application specifications, and scale-out replicas still use normal
ready/placement/resource admission. Explicit `ROUTE` bindings are logical-flow
path bindings for future transfers, not a claim that Darpan can reprogram an
arbitrary physical network without a driver. Explicit logical-flow routing and
scaled endpoints are deliberately non-composable until a separate per-replica
load-balancing/routing contract exists.

Generic `REPLICATE`, `ISOLATE`, and catch-all `CUSTOM` are not stable controller
actions. Service replication is `SCALE`; future data replication/quarantine or
namespaced custom capabilities require their own schema, validation, canonical
events/state, Real/Twin handlers, and tests.

## Runtime

`Session` is the execution center. It owns the clock, EventLog, StateStore,
validation/arbitration, observers, and one runtime backend. Physical and Twin
backends implement the same small backend contract. A physical cluster session
uses strict node-to-Agent mapping: system nodes without inventory executors are
invalid and never fall back to controller-local execution.

## Physical deployment and acceptance

Provisioning is outside the Agent runtime protocol and must remain auditable. Bootstrap
plans may contain rendered systemd units and remote commands, but never secret token values
or TLS private-key contents. Applying a bootstrap resolves controller-side token environment
variables only at execution time and transports secret environment-file contents over SSH
stdin rather than embedding them in remote command arguments. Physical-control capabilities
are granted only to inventory nodes that explicitly opt in.

A deployed host is not considered paper-ready merely because SSH or the Agent is reachable.
First-run acceptance must exercise the declared data plane and canonical long-running control
semantics on real Agent processes. Its deployment receipt binds the accepted cluster/system
input SHA256 values, exact Darpan release, active acceptance capabilities, and stable Physical
environment fingerprint. A Study may freeze that receipt as part of readiness identity; live
readiness must then reject a modified receipt, changed inputs, release mismatch, or environment
drift. Receipt evidence never replaces live readiness.

Discovery may propose observed node resources and environment labels, but it must not invent
physical topology links that cannot be inferred safely from host introspection. Human-reviewed
topology remains a frozen study input.

## Digital Twin

A Twin is built from model calibration, immutable snapshots, ordered scenario
modifications, virtual time, simulation, prediction uncertainty, and
comparison. A scenario does not mutate the snapshot from which it was forked.

Twin models have a lifecycle rather than a single `predict()` method:

- observe real or simulated events;
- advance through time;
- predict a query;
- snapshot and restore their state.

This supports time-evolving models such as thermal, battery, workload, and
failure dynamics. Default scheduling models also preserve named-resource
capacity and shared-link contention instead of assuming unlimited parallelism.

## Failure and recovery

- Application completion is terminal; recovery must not silently reopen an
  already-completed application.
- Finite-component retry is a bounded `ComponentSpec` execution policy, not an
  Action kind. Eligible failed attempts remain canonical facts, enter
  `component.retrying`, and return through normal ready/placement semantics before
  terminal dependency failure is propagated.
- Retry backoff is scheduled by Session using the runtime clock, so the same
  configuration is deterministic under Twin VirtualClock and managed under Real
  WallClock.
- Long-running `RESTART` and restart-style `MIGRATE` are explicit Actions for
  components that are already running; they do not resurrect terminal DAG work.
- Recovery liveness comes from ordinary event-driven feasibility re-evaluation;
  a ready component with no legal node waits rather than bypassing validation or
  forcing RL/MARL to choose an impossible action.

## Research extensions

Researchers extend Darpan through public protocols or adapters. A new research
idea should not require edits to Core. The current contracts cover policies,
telemetry, Twin models, metrics, analyzers, perturbations, executors,
simulation kernels, RL environments, optimization evaluators, and external
process policies.

## Reproducibility

Experiment seeds are executable configuration, not metadata-only fields. Recorded
artifacts retain raw canonical events, model/state snapshots, checksummed inputs,
and environment/Git provenance. Local experiment plugin source files are inputs,
not merely import strings. Calibrated model snapshots can be explicit checksummed
inputs to later runs, so adaptation can continue across process boundaries.

Paper suites freeze named experiments, scenarios, evidence, and recursive inputs
under a deterministic suite fingerprint. Campaign plans separately freeze job
configuration, matched seeds, statistics, suite identity, Physical node mapping,
and required-evidence coverage under a plan fingerprint. Execution must verify
configured suite/plan locks before starting work.

Study manifests form the orchestration identity above those two layers. A Study
fingerprint freezes its portable campaign reference/checksum, campaign-plan identity,
readiness policy, analysis policy, and research-question evidence requirements. Study
execution reuses CampaignRunner and its per-seed checkpoints; it must not introduce a
second experiment engine. Resuming after campaign completion may continue analysis
without rerunning Physical readiness or experiments. Study verification recursively
checks nested seals, planned seed completeness, and Physical restoration evidence.

Learning-efficiency campaigns may freeze explicit real-interaction, virtual-interaction,
wall-time, evidence-point, and threshold-reach budgets. A trace violating a declared
budget is invalid evidence for that job rather than a result that may be silently
aggregated.

Successful and failed experiment/campaign directories are sealed research
artifacts. Their manifest records schema/version identity, experiment/plan
fingerprints, per-file SHA256 values, and a content fingerprint; verification
must detect modification, removal, or untracked additions after sealing. Physical
cluster validation also records the Agent protocol/environment fingerprint so
hardware provenance is durable evidence rather than prose-only metadata.

Paired benchmark comparisons must use matched seeds and keep raw samples alongside
aggregate statistics. Layered fidelity measurements predict before observing/
calibrating the current sample, and contention/path metadata determines whether a
physical observation is eligible to train a baseline model.
