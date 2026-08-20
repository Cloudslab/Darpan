# Darpan architecture contract

Darpan freezes one canonical Continuum language across physical and digital
execution. The stable center is:

```text
Event -> EventLog -> Reducer -> ContinuumState -> Session -> Action -> Backend
                                                        /              \
                                                   Physical          Digital Twin
```

This center is intentionally smaller than the complete platform. A DRL
algorithm, optimizer, policy, benchmark, cluster transport, or Twin model is a
consumer or implementation of a public contract; none of them defines Darpan
Core.

## Canonical semantics

### Events are facts

Backends report runtime facts through canonical events. The Reducer is the only
place that turns those facts into `ContinuumState`. Controllers and research
algorithms consume state/events rather than mutating state directly.

Action execution has a complete lifecycle:

```text
action.requested
      |
      +-- action.rejected
      |
      +-- action.accepted -> action.started
                                |
                                +-- action.completed
                                +-- action.failed
```

`PolicyDispatcher` ignores action-lifecycle feedback by default and processes
new events through a serialized queue. This prevents a policy from recursively
reacting to the actions it just caused.

### Action vocabulary is not backend capability

`ActionKind` is the shared vocabulary of control intents. Real and Twin
backends currently implement `component.place`, `component.stop`,
`component.migrate`, and `component.restart`. Migration is restart-style movement
of a running long-lived component; restart is an in-place process restart of a
running long-lived component. Neither claims terminal-failure resurrection or
live process-memory migration. Unsupported kinds such as `network.route` are
rejected by `BackendCapabilityValidator`
before backend dispatch. `Session.supported_action_kinds` and
`Session.supports_action()` expose this distinction to extensions.

A new stable action is complete only when its validator, Real semantics, Twin
semantics, canonical events, state reduction, and contract tests agree.

## Application and workload boundary

An `ApplicationSpec` is a DAG of components and flows. Components may be finite
tasks or long-running services/streams. File flows can identify the produced
artifact and the consumer target path.

A `WorkloadSpec` adds timed arrivals. Arrival scheduling is runtime-neutral:
Twin uses its deterministic simulation queue and Real uses Session-managed
wall-clock tasks. Both produce the same application-submission events.

Adding an application should therefore mean providing configuration plus the
real executable/container, not adding a Darpan-internal application module.

## Physical runtime

`RealBackend` executes canonical actions through an executor boundary. The
local executor and remote Agent executor share the same placement contract.
For file flows, the backend stages inputs before allocating compute resources.
Concurrent placements pass through an asynchronous resource-capacity gate so
a batch validated against one state snapshot cannot overcommit the physical
node. Physical resource admission uses effective capacity: the latest physical
capacity measurement may shrink but never silently expand the system declaration.
Remote-to-remote artifacts use bounded chunks. Controller-streaming is
the safe fallback; an explicitly enabled source Agent can instead forward the
artifact directly to the target Agent using a short-lived, target-issued,
path/size-bound upload ticket. The target's long-lived token is never delegated
to the source. Docker components mount the same isolated artifact workspace used
by local/remote transfer. Direct-transfer events separate source-agent data-plane
duration from controller coordination overhead so only representative network
samples calibrate the Twin.

The Agent control plane supports authenticated/TLS transport, telemetry,
asynchronous start/wait/cancel, chunked artifacts, opt-in direct artifact
forwarding, and opt-in Agent-to-Agent network probes. A Session-managed cluster
monitor converts connectivity changes into canonical node offline/recovered
events. On node loss, `RealBackend` cancels compute and active artifact staging
whose data path uses that node, releases both its internal reservation and any
canonical allocation already emitted, and records terminal transfer/component
failure. Node recovery never silently restarts failed work at the backend. Finite
tasks/functions may separately declare a bounded application-level retry policy;
an eligible failed attempt becomes `component.retrying` and later `ready` before
application terminalization, so normal policy/validator placement chooses the
next node. Link probes emit canonical latency/bandwidth measurements from the
source physical node rather than measuring controller-to-node paths.

`darpan cluster validate` is the physical preflight boundary. It checks Agent
identity/protocol/feature handshake and telemetry, and can explicitly exercise
source-Agent network probes plus direct artifact transfer with SHA256 validation.
The preflight also compares declared CPU with observed host/cgroup capacity and
writes a durable report with checksummed inputs. `darpan cluster exercise` is the
active runtime drill: real Agent process placement -> in-place restart ->
restart-style migration -> stop, with canonical/physical handoff evidence. Neither
preflight nor the drill replaces the actual application benchmark campaign.


## Recovery contract

Finite-component retry and long-running restart/migration are deliberately
different contracts. `ComponentSpec.retry` is declarative execution policy for a
finite task/function: it has a bounded retry count, optional failure-kind filter,
and deterministic backoff. Each failed attempt remains in the EventLog. While an
eligible retry remains, Orchestrator emits `component.retrying`, increments the
attempt, schedules the next `component.ready` through Session/VirtualClock, and
does not propagate dependency failure or complete the application. Only the
exhausted final failure follows the existing terminal dependency/application
path. Application completion is therefore never reopened.

A retry returns through the ordinary placement boundary rather than bypassing
controllers. Built-in policies re-scan ready components after node/link/resource/
capacity recovery events, while RL/MARL surfaces a controlled decision only when
its action mask contains a legal placement. Real finite retries remove declared
output artifacts from the failed attempt before re-execution. `RESTART` and
restart-style `MIGRATE` remain explicit Actions for an already-running long-lived
component; they are not substitutes for finite-task retry.

Recovery evidence is canonical and durable: retry counts, recovered/exhausted
components, retry downtime, and per-seed scenario recovery summaries are derived
from the same event log. Offline trace fidelity compares retry downtime across
Real/Twin without inventing a separate recovery simulator.

## Digital Twin runtime

`TwinBackend` executes the same actions against a virtual clock and replaceable
`SimulationKernel`. Its default models cover execution time, artifact size,
network delay, and queue delay. Models expose prediction uncertainty and are
snapshot/restorable.

Physical observations calibrate the Twin through the same event stream. A
`DigitalTwin` can attach before execution or attach later and replay Session
history. `TwinSnapshot` captures state/model/RNG state; `Scenario` applies
ordered modifications and injected events before simulation. Portable Twin
experiments can also reference a `scenario:` plan whose timed node offline/
recovery, link down/change, and measurement events are injected through the same
canonical Event -> Reducer path. Runtime node/link faults cancel affected compute
or transfers and invalidate already-queued future success events rather than
letting failed work complete later.

For a declared artifact flow, Twin emits the same transfer-start/complete event
kinds as Real and combines artifact-size and network uncertainty when predicting
transfer risk. Named resources use strict reservation by default. CPU resources
that explicitly declare `attributes.scheduling: fair` use dynamic max-min
sharing while strict secondary resources (for example memory) remain reserved.
Live physical CPU-capacity telemetry can reduce the effective shareable
capacity. Network predictions expose route link IDs and an event-driven max-min
bandwidth scheduler rebalances active flows as contenders join/leave or live
bandwidth changes; disjoint paths proceed concurrently. Canonical link
measurements override static latency and bandwidth so physical telemetry changes
subsequent predictions.

## Research adapters

### RL

`RLProblem` defines only the learning-facing boundary: observation, action
mapping, reward, action-space size, and the subset of ready decisions controlled
by RL. Uncontrolled decisions can be delegated to a normal Darpan `Policy`.
`DarpanEnv` and `MultiAgentDarpanEnv` run the same problem against Real or Twin.
The optional `darpan-rl` package implements learning algorithms and depends on
Darpan; Darpan Core never depends on it. Learning/sample-efficiency evidence is
kept algorithm-neutral: cumulative points record observed quality, real and
virtual interaction counts, and wall time. The official trainer can export that
format, but external agents or optimizers can produce exactly the same evidence
without importing `darpan-rl`. Twin rollouts are counted as virtual interactions,
not silently promoted into physical policy-quality observations.

### Optimization and external controllers

Optimization adapters and process/plugin policies ultimately produce canonical
Darpan actions. They do not receive a private simulator API. This keeps the
algorithm/runtime boundary shared across research methods.

## Six research capabilities

Robust, Trustworthy, Adaptive, Proactive, Explainable, and Explorable behavior
are capabilities built from shared primitives rather than separate framework
stacks:

- trustworthy: expected-vs-observed `Comparison`;
- adaptive: calibration before/after comparison;
- proactive: future `Prediction` with horizon/uncertainty;
- explainable: canonical action/event provenance;
- explorable: snapshot/scenario simulation;
- robust: repeated perturbation scenarios and worst-case degradation.

The `darpan.experiment` assessment helpers intentionally stay metric-agnostic so
paper-specific success criteria do not leak into Core.

## Extension test

A proposed abstraction is not considered complete because a class or enum name
exists. It must survive these user-level tests:

1. a researcher can bring a new DRL/reward without editing Darpan Core;
2. a researcher can add a new application/workload as external configuration;
3. the same controller boundary can move from Twin to a physical cluster;
4. physical observations can improve subsequent Twin predictions;
5. a new backend capability fails explicitly where unsupported and has matching
   Real/Twin semantics where supported;
6. a paper artifact can reproduce the experiment from portable configuration.

These tests are the reason Darpan keeps a small stable center and pushes
research-specific behavior to adapters, models, policies, metrics, and
executors.

## Reproducibility and benchmark boundary

Portable experiment specifications own seed/repeat/output policy. Recorded runs
include canonical events, final state, model snapshots, resolved inputs with
SHA256 checksums, and environment/Git provenance. Paired benchmarks run baseline
and candidate experiments with matched seeds and retain raw per-seed run evidence
plus a bootstrap confidence interval. Benchmark performance metrics reject failed
runs by default; explicit success-rate studies can opt out, preventing a fast
failure from being interpreted as a latency win.

A `CampaignSpec` composes paired, trace-fidelity, capability, scenario, and
learning-evidence jobs without creating a second execution model. It checkpoints
a durable campaign state after every job, recursively captures referenced inputs,
and preserves raw experiment artifacts. Scenario campaigns report both paired
metric degradation and success rates. Learning campaigns compare threshold reach
rate, real interactions to threshold, wall time, virtual interactions, and final
quality on matched seeds. The campaign layer is evidence orchestration only; it
does not move paper-specific semantics into Core.

Twin execution fidelity is measured out-of-sample: a prediction is captured
before the physical completion event is used to calibrate the model. Contention-
marked physical samples remain in fidelity artifacts but are excluded from solo
execution/network baseline calibration to avoid learning contention twice. Model
snapshots are durable inputs: a later Real fidelity or Twin experiment can warm-
start from `models.json`, a prior `fidelity.json`, or
`TwinSnapshot.model_states`.
