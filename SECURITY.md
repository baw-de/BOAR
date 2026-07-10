# Security Policy

## Supported Versions

Only the latest released version of BOAR receives security fixes.

| Version | Supported |
|---------|-----------|
| 1.x (latest) | ✅ |
| < 1.0 | ❌ |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

If you discover a security vulnerability in BOAR, please report it responsibly:

1. **Email** the maintainers at the address listed in `pyproject.toml` (or open a [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability) via the **"Report a vulnerability"** button on the repository's Security tab).
2. Include as much detail as possible:
   - Description of the vulnerability and its potential impact.
   - Steps to reproduce or a proof-of-concept (if available).
   - Affected version(s).
   - Any suggested mitigations.

We will acknowledge receipt within **5 business days** and aim to provide a fix or mitigation within **90 days**, depending on severity.

---

## Scope

BOAR is a scientific calibration tool intended to be run in trusted, local or HPC environments.  
The following areas are most relevant from a security perspective:

| Area | Notes |
|------|-------|
| **File I/O** | BOAR reads user-supplied YAML configs and HDF5 model output files. Malformed or malicious input files could cause unexpected behaviour. |
| **Subprocess execution** | BOAR may invoke external hydrodynamic model executables (e.g., BASEMENT). Ensure model binaries and configuration paths come from trusted sources. |
| **Dependencies** | Third-party packages (`numpy`, `torch`, `optuna`, etc.) are audited via `pip-audit` in CI. Keep your environment up to date. |

### Out of Scope

- Vulnerabilities in the underlying hydrodynamic modelling software (e.g., BASEMENT).
- Issues arising from running BOAR with untrusted user-supplied model executables.
- General Python runtime or OS-level vulnerabilities.

---

## Security Practices in This Repository

- **Dependency auditing**: `pip-audit` is run in CI on every push.
- **Static analysis**: `bandit` scans the source code for common security issues on every push.
- **Minimal permissions**: BOAR does not require network access or elevated system privileges during normal operation.

---

## Disclosure Policy

Once a fix is released, we will:

1. Publish a [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories) with the CVE (if applicable).
2. Credit the reporter (unless they prefer to remain anonymous).
3. Tag a new release with the patch.

---

*Thank you for helping keep BOAR and its users safe.*
