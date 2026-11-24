## Summary

<!-- What changed and why -->

## Type of change

- [ ] Bug fix
- [ ] New provider / adapter
- [ ] Documentation
- [ ] Tests
- [ ] Examples
- [ ] Other (describe)

## Checklist

- [ ] Tests added or updated (`pytest --cov=nvs_runtime` passes)
- [ ] Docs updated if user-facing behavior changed
- [ ] Examples updated when the demonstrated flow changed
- [ ] Backwards compatibility preserved for the Runtime API  
      (`discover` / `plan` / `invoke` / `observe` / `session` / `checkpoint` / `metrics`)  
      — or breaking changes are called out explicitly
- [ ] Provider-specific code stays under `providers/`
- [ ] Offline / deterministic (no API keys, no live network required)
