# Governance

NVS Runtime is governed lightly so the Runtime API stays stable while the
provider ecosystem can evolve.

## Project Vision

NVS Runtime is a **provider-agnostic execution substrate** for layer-oriented
AI systems.

- Applications depend on the Runtime API, never on a specific provider.
- MCP, HTTP, gRPC, Local, Kernel, HEKB, and future adapters are peers.
- The runtime must remain usable offline and testable without external APIs.
- We optimize for long-term API stability over short-term feature density.

## Decision Process

1. **Routine changes** (docs, tests, bug fixes, non-breaking provider work)  
   → discussed in issues/PRs; merged by a maintainer after review.
2. **User-facing or architectural changes**  
   → open an issue first; prefer consensus among active maintainers.
3. **Breaking Runtime API changes**  
   → require an RFC and an explicit release note; reserved for major versions
   unless a security fix demands otherwise.

When consensus is unclear, maintainers decide. Rationale should be recorded in
the issue, PR, or RFC.

## RFC Process

Use an RFC for proposals that affect:

- the stable Runtime API (`discover` / `plan` / `invoke` / `observe` /
  `session` / `checkpoint` / `metrics`)
- core architecture (planner, capability graph, governance, transport)
- cross-cutting provider contracts

Suggested flow:

1. Open a Discussion (category **RFC**) or an issue labeled `rfc`.
2. Describe problem, proposal, alternatives, and compatibility impact.
3. Allow a short comment period (typically ≥7 days for non-urgent RFCs).
4. A maintainer accepts, defers, or declines with a brief public note.

Small provider additions do **not** need an RFC if they preserve contracts.

## Maintainers

| Role | Responsibility |
|------|----------------|
| Maintainers | Merge rights, releases, security coordination, RFC decisions |
| Contributors | Patches, docs, providers, reviews via pull requests |

Current maintainers are listed in the GitHub repository settings and may be
mirrored in `CODEOWNERS` when that file is introduced.

Maintainers are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md)
and keep the Runtime API trustworthy for downstream users.

## Release Policy

| Channel | Policy |
|---------|--------|
| **SemVer** | `MAJOR.MINOR.PATCH` — Runtime API breaks require `MAJOR` |
| **v1.x** | Additive, backwards-compatible changes preferred |
| **Patches** | Bug fixes and security fixes; no intentional API breaks |
| **Tags** | Git tags `vX.Y.Z` with GitHub Releases and notes |
| **Security** | Coordinated disclosure per [SECURITY.md](SECURITY.md) |

Pre-release tags (e.g. `v1.1.0-rc.1`) may be used for soak testing.
Experimental provider features should not break the stable Runtime API.
