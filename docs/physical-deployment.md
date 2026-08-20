# Physical deployment and first-run workflow

Darpan's deployment kit is intentionally split into provisioning, active acceptance, and
study execution. A host being reachable over SSH is not treated as proof that it is ready
for a Physical Study, and an old acceptance result is not treated as proof that the host
has not changed since then.

## 1. Prepare the cluster inventory

Each node needs an Agent endpoint and SSH destination. Secrets stay in controller
environment variables rather than in YAML:

```yaml
nodes:
  - id: edge-1
    host: 192.168.1.21
    port: 8765
    ssh_user: ubuntu
    ssh_port: 22
    token_env: DARPAN_EDGE_1_TOKEN
    tls_ca: certs/ca.pem
    tls_server_name: edge-1.example.org
    network_probe: true
    artifact_forward: true
    physical_control: true
    control_interfaces: [eth1]
```

Set the referenced token variables on the controller before `bootstrap --apply` or any
Agent RPC command. If TLS is used, the controller keeps the CA certificate while the
bootstrap TLS source directory contains `<node-id>.crt` and `<node-id>.key` for each
Agent.

## 2. Generate and inspect a bootstrap plan

Build the Core wheel and generate an auditable plan without changing a remote machine:

```bash
python -m pip wheel . --no-deps --no-build-isolation -w wheelhouse

darpan cluster bootstrap \
  --cluster configs/systems/cluster.example.yaml \
  --wheel wheelhouse/darpan_continuum-0.2.0.dev27-py3-none-any.whl \
  --tls-source-dir certs/agents \
  --output results/bootstrap-plan
```

The plan renders the systemd unit and remote commands. It never stores the token value or
TLS private-key contents. Physical-control nodes receive `CAP_NET_ADMIN`; other Agents do
not. Docker-group membership is high privilege and is never granted unless
`--grant-docker-group` is explicitly requested.

## 3. Apply provisioning

Use a new output directory for the applied deployment:

```bash
darpan cluster bootstrap \
  --cluster configs/systems/cluster.example.yaml \
  --wheel wheelhouse/darpan_continuum-0.2.0.dev27-py3-none-any.whl \
  --tls-source-dir certs/agents \
  --apply \
  --output results/bootstrap-apply
```

The Agent is installed into a dedicated virtual environment and launched as a systemd
service with a persistent workspace under `/var/lib/darpan`. The shared token is sent over
SSH stdin to a mode-0600 environment file; it is not embedded in the SSH command line,
bootstrap plan, or durable artifact. TLS private keys are uploaded directly and installed
mode 0600; they are not copied into the controller-side bootstrap artifact.

## 4. Run first-run acceptance

The recommended next command is the combined first-run workflow:

```bash
darpan cluster first-run \
  --cluster configs/systems/cluster.example.yaml \
  --system configs/systems/edge_fog_cloud.yaml \
  --source edge-1 \
  --target fog-1 \
  --exercise-physical-control \
  --output results/cluster-first-run
```

It performs, in order:

1. SSH prerequisite checks (Python, systemd, disk, optional Docker, and configured Linux
   control targets);
2. deployed Agent/protocol/clock/system validation;
3. live Agent discovery and environment fingerprinting;
4. Agent-to-Agent network probe and direct artifact transfer;
5. PLACE, RESTART, MIGRATE, SCALE 1→2→1, and STOP physical lifecycle checks;
6. optional workload-plane disable/restore when `--exercise-physical-control` is set.

`tc` link mutation is intentionally a separate opt-in:

```bash
--exercise-link-control
```

because it changes an allow-listed physical interface. The first-run artifact is sealed and
contains `deployment-receipt.json`. A successful receipt binds the accepted cluster and
system SHA256 values to the live Physical environment fingerprint.

If SSH provisioning is managed by another system, `--skip-ssh-check` is available only as
an explicit override. The resulting artifact records that the SSH layer was skipped.

## 5. Review discovery before freezing topology

`first-run` includes discovery output. It can also be run independently:

```bash
darpan cluster discover \
  --cluster configs/systems/cluster.example.yaml \
  --output results/discovery
```

Darpan can infer host resources and environment metadata, but it deliberately does not
invent physical topology links. Review `cluster.discovered.yaml` and
`system.discovered.yaml`, add the real topology/control labels, and freeze those reviewed
files through the normal suite/campaign/study locks.

## 6. Bind the real Study to the deployment receipt

Use the explicit binding command rather than hand-editing the frozen Study:

```bash
darpan study bind-deployment my-physical-study-template.yaml \
  results/cluster-first-run \
  --output frozen/my-physical-study.yaml \
  --lock-output frozen/my-physical-study.lock.json
```

The source template is never modified. Binding first verifies the sealed acceptance/first-run
artifact, requires the Study to be Physical with readiness enabled, and requires the frozen
campaign to expose exactly one cluster/system SHA256 pair matching the deployment receipt.
It then writes a fresh Study whose `readiness.acceptance_receipt` points at that artifact and
a matching immutable Study lock.

The receipt artifact manifest fingerprint and deployment fingerprint become part of the
Study identity. Before an incomplete Physical campaign runs, live readiness is performed
again. Darpan rejects the Study if:

- the receipt was modified or is not sealed;
- it was not marked `ready_for_study`;
- the current cluster or system file SHA256 differs from the accepted inputs; or
- the live Physical environment fingerprint differs from the accepted environment.

The receipt therefore proves prior active acceptance while live readiness proves that the
accepted environment still matches now.

## 7. Run the Physical Study

```bash
darpan study run frozen/my-physical-study.yaml --output results/physical-study
```

Long campaigns retain the existing per-seed resume semantics:

```bash
darpan study run frozen/my-physical-study.yaml \
  --output results/physical-study \
  --resume
```

If the campaign already completed and only analysis was interrupted, resume does not touch
Physical hardware again.

## Safety boundaries

- Built-in Physical `node.offline` is Darpan workload-plane unavailability, not host
  power-off or IPMI/BMC failure.
- Linux `tc` control is interface-scoped, not per-flow.
- The Linux route driver is host-route control, not SDN/OpenFlow per-flow routing.
- Agent physical control is restricted; it does not expose arbitrary shell execution.
- Existing non-trivial qdiscs are refused by default when restoration cannot be verified.
- The Agent control lease improves controller-crash rollback, but cannot restore a host
  that itself crashed or lost management/root access.
- Every deployment/acceptance output directory is fresh-only so evidence from separate
  runs cannot be silently mixed.
