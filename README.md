# Data Planning Agent

An MCP (Model Context Protocol) agent that transforms high-level business intents into structured **Data Product Requirement Prompts** (Data PRPs) through AI-powered conversational refinement.

## Overview

The Data Planning Agent is the first component in a multi-agent system for automated Business Intelligence dashboard generation. It helps data scientists and analysts gather comprehensive requirements by:

1. **Starting** with a vague business intent
2. **Refining** through AI-guided clarifying questions
3. **Generating** a structured, machine-readable Data PRP document

The output Data PRP serves as input for the **Data Discovery Agent**, enabling automated data source identification and analysis.

## Features

- 🤖 **AI-Powered Conversations**: Uses Gemini 2.5 Pro for intelligent requirement gathering
- ❓ **Smart Questioning**: Asks up to 4 focused questions at a time, biased toward multiple choice for efficiency
- 📋 **Structured Output**: Generates standardized Data PRP markdown documents
- 💾 **Flexible Storage**: Supports both GCS (`gs://`) and local file paths
- 🎨 **Organizational Context**: Load custom context files to tailor agent behavior to your organization
- 🔌 **MCP Integration**: Full MCP server implementation (stdio + HTTP transports)
- 🖥️ **Interactive CLI**: Test conversations directly from the command line
- 🎯 **Cursor Compatible**: Works seamlessly as a Cursor MCP server

## Installation

### Prerequisites

- Python 3.10 or higher
- Poetry for dependency management
- Gemini API key

### Setup

1. Clone the repository:
```bash
cd /home/user/git/data-planning-agent
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a `.env` file from the example:
```bash
cp .env.example .env
```

4. Configure your environment variables in `.env`:
```bash
# Required
GEMINI_API_KEY=your-gemini-api-key-here

# Optional (with defaults)
GEMINI_MODEL=gemini-2.5-pro
OUTPUT_DIR=./output
MCP_TRANSPORT=stdio
LOG_LEVEL=INFO
```

## Usage

### Interactive CLI Mode

The easiest way to test the Planning Agent:

```bash
poetry run planning-agent
```

This launches an interactive session that guides you through:
1. Entering your initial business intent
2. Answering clarifying questions
3. Generating and saving the final Data PRP

### MCP Server Mode (for Cursor Integration)

Run as an MCP server for integration with Cursor:

```bash
# Stdio transport (default)
poetry run python -m data_planning_agent.mcp

# HTTP transport
MCP_TRANSPORT=http poetry run python -m data_planning_agent.mcp
```

### Using with Cursor

Add this configuration to your `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "data-planning-agent": {
      "command": "poetry",
      "args": ["run", "python", "-m", "data_planning_agent.mcp"],
      "cwd": "/home/user/git/data-planning-agent",
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here",
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

Then use these MCP tools in Cursor:

#### 1. `start_planning_session`

Start a new planning session:

```json
{
  "initial_intent": "We want to provide the merchandising team insights into trending items in region 7"
}
```

Returns a session ID and initial clarifying questions.

#### 2. `continue_conversation`

Continue the conversation with responses:

```json
{
  "session_id": "your-session-id",
  "user_response": "a) Regional managers, they need both summary and detail"
}
```

Returns follow-up questions or completion notification.

#### 3. `generate_data_prp`

Generate the final Data PRP:

```json
{
  "session_id": "your-session-id",
  "output_path": "gs://my-bucket/planning/data_prp.md",
  "save_to_file": true
}
```

Returns the generated Data PRP markdown and file location.

## Example Conversation Flow

```
User: "We want to provide the merchandising team insights into trending items in region 7"

Agent: Based on your intent, I have a few questions:

1. What is the primary audience for this analysis?
   a) Executives (high-level summary)
   b) Regional managers (summary + detail)
   c) Data analysts (detailed data)
   d) Other (please specify)

2. What key metrics define "trending" for your use case?
   a) Unit sales volume
   b) Revenue growth
   c) Profit margin
   d) Multiple metrics (please specify)

3. What time frame should we analyze?
   a) Last 4 weeks
   b) Last 8 weeks
   c) Last quarter
   d) Custom period (please specify)

4. Do you need comparisons to previous periods?
   a) Yes, week-over-week
   b) Yes, year-over-year
   c) Yes, both
   d) No comparisons needed

User: "b) Regional managers
       a) Unit sales volume
       b) Last 8 weeks  
       a) Yes, week-over-week"

Agent: [Asks follow-up questions or generates Data PRP]
```

## Data PRP Output Format

The generated Data PRP follows this structure:

```markdown
# Data Product Requirement Prompt

## 1. Executive Summary

* **Objective:** [One-sentence business goal]
* **Target Audience:** [Who will use this]
* **Key Question:** [Primary question to answer]

## 2. Business Context

[Detailed paragraph explaining the scenario and decisions to be made]

## 3. Data Requirements

### 3.1. Key Metrics

* [Metric 1]
* [Metric 2]

### 3.2. Dimensions & Breakdowns

* [Dimension 1]
* [Dimension 2]

### 3.3. Filters

* [Filter 1]
* [Filter 2]

## 4. Success Criteria

* **Primary Metric:** [Main success indicator]
* **Timeline:** [Delivery expectations]
```

## Organizational Context

The Planning Agent can be customized to your organization by loading context files that influence all AI interactions.

### What is Organizational Context?

Context files are markdown documents that provide the AI with:
- Company-specific terminology and standards
- Standard operating procedures (SOPs)
- Data governance policies
- Technical constraints
- Communication preferences

### How to Use Context

1. **Create a context directory** (local or GCS):
   ```bash
   mkdir ./context
   ```

2. **Add markdown files** with your organizational knowledge:
   ```bash
   # context/01_organization.md
   # context/02_sop.md
   # context/03_constraints.md
   ```

3. **Configure the agent** to use your context:
   ```bash
   # .env
   CONTEXT_DIR=./context
   # or for GCS:
   # CONTEXT_DIR=gs://my-bucket/planning-context/
   ```

4. **Files are loaded automatically** when the agent starts

### Example Context Files

See the `context.example/` directory for real examples:

- **01_organization.md**: Organizational background, team structure, communication style
- **02_sop.md**: Standard operating procedures, terminology standards, data governance
- **03_constraints.md**: Technical constraints, preferred analysis patterns, budget considerations

### Benefits

- **Consistency**: Agent uses your terminology and follows your SOPs
- **Governance**: Automatically applies your data governance policies
- **Efficiency**: No need to repeat organizational context in every conversation
- **Flexibility**: Update context files without changing code

### Context Behavior

- Context is **prepended to all AI prompts** (initial questions, follow-ups, PRP generation)
- Context is **hidden from users** - it silently guides agent behavior
- Context is **optional** - agent works normally without it
- Multiple files are **concatenated alphabetically**
- Supports both **local** and **GCS** storage

## Configuration

All configuration is managed through environment variables. See `.env.example` for the complete list:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini API key (required) | - |
| `GEMINI_MODEL` | Gemini model to use | `gemini-2.5-pro` |
| `OUTPUT_DIR` | Default output directory | `./output` |
| `CONTEXT_DIR` | Context directory (local or GCS) | None |
| `MCP_TRANSPORT` | Transport mode (`stdio` or `http`) | `stdio` |
| `MCP_HOST` | HTTP server host | `0.0.0.0` |
| `MCP_PORT` | HTTP server port | `8080` |
| `MAX_CONVERSATION_TURNS` | Max conversation turns | `10` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Architecture

### Components

- **MCP Server** (`src/data_planning_agent/mcp/`)
  - Stdio and HTTP transports
  - JSON-RPC 2.0 protocol
  - SSE support for real-time updates

- **Clients** (`src/data_planning_agent/clients/`)
  - `GeminiClient`: Gemini API wrapper for conversations
  - `StorageClient`: GCS and local file I/O

- **Core Logic** (`src/data_planning_agent/core/`)
  - `ConversationManager`: Session state management
  - `RequirementRefiner`: Conversation orchestration
  - `PRPGenerator`: Data PRP markdown generation

- **Models** (`src/data_planning_agent/models/`)
  - `PlanningSession`: Session data model
  - `DataProductRequirementPrompt`: PRP schema

- **CLI** (`src/data_planning_agent/cli/`)
  - Interactive command-line interface

### Integration with Data Discovery Agent

```
┌─────────────────────┐
│  Planning Agent     │  1. Gathers requirements
│  (This repo)        │     through conversation
└──────────┬──────────┘
           │
           │ Data PRP.md
           ▼
┌─────────────────────┐
│ Data Discovery      │  2. Searches for relevant
│ Agent               │     datasets using PRP
└──────────┬──────────┘
           │
           │ Discovered datasets
           ▼
┌─────────────────────┐
│ Query Generation    │  3. Generates SQL queries
│ Agent               │     for analysis
└─────────────────────┘
```

## Testing

Run tests with pytest:

```bash
# All tests
poetry run pytest

# Unit tests only
poetry run pytest tests/unit/

# With coverage
poetry run pytest --cov=data_planning_agent --cov-report=html
```

## Development

### Code Quality

Format code with Black:
```bash
poetry run black src/ tests/
```

Lint with Ruff:
```bash
poetry run ruff check src/ tests/
```

### Project Structure

```
data-planning-agent/
├── src/data_planning_agent/
│   ├── mcp/              # MCP server implementation
│   ├── clients/          # External service clients
│   ├── core/             # Business logic
│   ├── models/           # Data models
│   └── cli/              # Command-line interface
├── tests/                # Test suite
├── pyproject.toml        # Poetry configuration
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Troubleshooting

### Common Issues

**Issue**: `GEMINI_API_KEY not set`
- **Solution**: Ensure your `.env` file contains a valid Gemini API key

**Issue**: Session timeout or max turns reached
- **Solution**: Increase `MAX_CONVERSATION_TURNS` in `.env`

**Issue**: GCS write permission denied
- **Solution**: Ensure your GCP credentials have write access to the bucket

**Issue**: Cursor can't connect to MCP server
- **Solution**: Check that `MCP_TRANSPORT=stdio` and the `cwd` path is correct

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Related Projects

- [Data Discovery Agent](https://github.com/your-org/data-discovery-agent) - Discovers relevant datasets
- [Query Generation Agent](https://github.com/your-org/query-generation-agent) - Generates SQL queries
- [Data Discovery Infrastructure](https://github.com/your-org/data-discovery-infrastructure-gcp) - GCP infrastructure

## Support

For issues, questions, or contributions, please open an issue on GitHub.

