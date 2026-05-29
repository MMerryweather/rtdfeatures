# Security Policy

## Supported Versions

`rtdfeatures` follows a latest-stable support policy:

- The most recent stable release receives security fixes.
- Older releases may not receive security updates.

## Reporting a Vulnerability

Please do not open public issues for suspected vulnerabilities.

- Private report email: **REDACTED_EMAIL**
- Include: affected version, reproduction steps, impact, and any known mitigations.

## Response Expectations

- Acknowledgement target: within 5 business days
- Triage target: initial severity/risk assessment after acknowledgement
- Remediation and disclosure: coordinated disclosure after a fix is available

## Security Baseline

This repository maintains a minimum public-release security baseline:

- Static security checks are run where supported by the active repository plan
- Dependency audit runs in CI via `pip-audit` in `.github/workflows/ci.yml`

Repository-level secret scanning and push protection should be enabled in GitHub settings when available for the repository plan.
