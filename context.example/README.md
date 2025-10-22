# Example Organizational Context

This directory contains example context files that demonstrate how to customize the Data Planning Agent for your organization.

## Purpose

Context files are markdown documents that provide the AI agent with organizational knowledge:

- Company-specific terminology
- Standard operating procedures
- Data governance policies
- Technical constraints
- Communication preferences

## Usage

1. Copy this directory to your own location (local or GCS)
2. Modify the example files or create your own
3. Set `CONTEXT_DIR` environment variable to point to your context directory
4. Context will be automatically loaded when the agent starts

## File Naming

Files are loaded in alphabetical order and concatenated. Use numbered prefixes to control order:

- `01_organization.md` - Organizational context (loaded first)
- `02_sop.md` - Standard operating procedures
- `03_constraints.md` - Technical and business constraints
- etc.

## Example Configuration

### Local Context

```bash
# .env
CONTEXT_DIR=./context
```

### GCS Context

```bash
# .env
CONTEXT_DIR=gs://my-company-bucket/planning-agent-context/
```

## File Structure

Each markdown file can contain any free-form content. The agent will receive all context before every prompt.

**Recommended sections**:

- Organizational background
- Team structure
- Communication style
- Domain-specific terminology
- Standard operating procedures
- Data governance policies
- Technical constraints
- Preferred analysis patterns

## Tips

- Keep context files focused and concise
- Update context files as policies change
- Use clear headers and formatting
- Include definitions for domain-specific terms
- Document any "always" or "never" rules
- Specify preferred terminology

## Effects

Context influences all agent interactions:

- **Initial questions**: Tailored to your domain and constraints
- **Follow-up questions**: Consistent with your terminology
- **Data PRP generation**: Aligned with your standards and requirements

Context is **not** shown to end users - it silently guides the agent's behavior.

