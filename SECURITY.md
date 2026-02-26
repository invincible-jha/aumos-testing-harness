# Security Policy

## Reporting a Vulnerability

The AumOS platform team takes security vulnerabilities seriously. We appreciate your
efforts to responsibly disclose your findings.

**Please do NOT report security vulnerabilities through public GitHub issues.**

### How to Report

Email your findings to: **security@aumos.io**

Include in your report:
- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any proof-of-concept code (if applicable)
- Your recommended fix (if you have one)

You will receive an acknowledgment within **48 hours** and a detailed response
within **5 business days**.

## Scope

The following are in scope for security reports in this repository:

- Authentication and authorization bypass on evaluation endpoints
- Tenant isolation violations (cross-tenant test result access)
- Prompt injection in red-team tooling that escapes to the host system
- SQL injection or other injection attacks
- Remote code execution via malicious test suite configurations
- Sensitive data exposure (API keys, tenant data, model outputs with PII)
- SSRF via configurable target endpoints in red-team assessments
- Privilege escalation in the evaluation runner

The following are out of scope:

- Denial of service attacks
- Vulnerabilities in third-party evaluation frameworks (report to Garak/Giskard upstream)
- Issues in repositories not owned by AumOS Enterprise
- Social engineering of AumOS staff

## Special Considerations for This Repository

Because `aumos-testing-harness` interacts with external AI models and runs attack probes,
there are additional security considerations:

1. **Red-team target endpoints** are always configured by authenticated tenants — never
   allow unauthenticated callers to specify arbitrary target URLs (SSRF risk).
2. **Evaluation test cases** may contain adversarial inputs — sanitize before logging.
3. **Red-team reports** may contain successful attack prompts — these are stored encrypted
   and never logged in plaintext.
4. **LLM API keys** in `AUMOS_TESTHARNESS_OPENAI_API_KEY` must never appear in logs,
   error messages, or API responses.

## Response Timeline

| Stage | Timeline |
|-------|----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 5 business days |
| Status update | Every 7 days during investigation |
| Fix deployment (critical) | Within 7 days of confirmation |
| Fix deployment (high) | Within 30 days of confirmation |
| Fix deployment (medium/low) | Next scheduled release |

## Disclosure Policy

- We follow a **90-day coordinated disclosure** policy
- We will notify you when the fix is deployed
- We will credit you in our release notes (unless you prefer anonymity)
- We ask that you do not publicly disclose the vulnerability until we have released a fix

## Security Best Practices for Contributors

When contributing to this repository:

1. Never commit secrets, API keys, or credentials (even test credentials)
2. Use parameterized queries — never string concatenation in SQL
3. Validate all inputs at system boundaries using Pydantic
4. Never log sensitive data (tokens, passwords, PII, raw LLM outputs)
5. Never allow user-supplied URLs in red-team targets without allowlist validation
6. Check dependency licenses and security advisories before adding packages
7. Report any security issues you discover, even if you are not sure of the impact
