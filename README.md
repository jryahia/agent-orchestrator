# Agent Orchestrator

A **Multi-Agent Workflow System** with a master orchestrator agent that breaks down goals into subtasks, delegates to specialized worker agents, collects results, and delivers final output.

## Features

- 🧠 **Master Orchestrator** — takes a goal, plans subtasks, assigns to workers, tracks progress, compiles final result
- 🤖 **Worker Agents:**
  - **Researcher** — web search + content extraction (httpx + BeautifulSoup)
  - **Writer** — summarizes and formats content into structured output
  - **Reviewer** — checks quality, suggests improvements
  - **Deliverer** — sends result via Telegram/file
- 🔗 **LLM Integration** — each agent uses OpenAI/DeepSeek/OpenRouter via the openai library
- 💾 **Task Memory** — SQLite stores all workflows, subtasks, and agent outputs
- 📊 **Progress Tracking** — each subtask shows its status (pending → running → completed/failed)
- 🔄 **Retry Logic** — failed subtasks retry up to 3 times
- ⛓️ **Max Depth** — prevents infinite delegation (max 3 levels)
- 🖥️ **CLI Interface** — run workflows from the command line
- 🌐 **API Server** — FastAPI-based REST API with Swagger docs
- 📝 **Output Formats** — markdown, JSON, text file, Telegram message

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jryahia/agent-orchestrator.git
cd agent-orchestrator

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Set your LLM API key (supports OpenAI, DeepSeek, OpenRouter):

```bash
export LLM_API_KEY="your-api-key-here"
export LLM_BASE_URL="https://api.openai.com/v1"  # or DeepSeek/OpenRouter endpoint
export LLM_MODEL="gpt-4o-mini"                    # or your preferred model
```

### CLI Usage

```bash
# Run a single workflow
python main.py "Research Top 5 AI startups and write a summary report"

# Run with specific agents
python main.py "Find latest Python features and summarize" --agents researcher writer

# Run multiple goals from a JSON file
python main.py --file examples/sample_goal.json

# Parallel execution
python main.py --file examples/sample_goal.json --parallel

# Dry run (test without real LLM calls)
python main.py "Research Python 3.14 features" --dry-run

# Save output to a file
python main.py "Research AI trends" --output-file my_report
```

### API Server

```bash
# Start the API server
python main.py --api

# Or directly
python api.py
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

#### API Endpoints

```bash
# Create a workflow
curl -X POST "http://localhost:8000/api/v1/workflows" \
  -H "Content-Type: application/json" \
  -d '{"goal": "Research top 5 AI startups", "agents": ["researcher", "writer"]}'

# Get workflow status
curl "http://localhost:8000/api/v1/workflows/{workflow_id}"

# Get workflow result
curl "http://localhost:8000/api/v1/workflows/{workflow_id}/result"

# Get workflow progress
curl "http://localhost:8000/api/v1/workflows/{workflow_id}/progress"

# List all workflows
curl "http://localhost:8000/api/v1/workflows"

# Delete a workflow
curl -X DELETE "http://localhost:8000/api/v1/workflows/{workflow_id}"

# Health check
curl "http://localhost:8000/health"
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Build only
docker build -t agent-orchestrator .
docker run -p 8000:8000 -e LLM_API_KEY=your-key agent-orchestrator
```

## Project Structure

```
agent-orchestrator/
├── main.py                    # CLI entry point
├── api.py                     # API server entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── planner.py         # Break goal into subtasks
│   │   ├── dispatcher.py      # Assign tasks to agents
│   │   └── tracker.py         # Track progress + retries
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py            # Base agent class
│   │   ├── researcher.py      # Web search + extract
│   │   ├── writer.py          # Summarize + format
│   │   ├── reviewer.py        # Quality check
│   │   └── deliverer.py       # Send result
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py          # Unified LLM client
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── task_store.py      # SQLite storage
│   │   └── models.py          # Pydantic schemas
│   └── server/
│       ├── __init__.py
│       ├── api.py             # FastAPI app
│       └── routes.py          # API routes
├── tests/
│   ├── __init__.py
│   ├── test_planner.py        # Planner tests
│   └── test_dispatcher.py     # Dispatcher tests
└── examples/
    └── sample_goal.json       # Example goals file
```

## How It Works

1. **Input**: A goal (e.g., "Research Top 5 AI startups and write a summary")
2. **Planning**: The Planner uses the LLM to decompose the goal into subtasks
3. **Dispatching**: The Dispatcher assigns each subtask to the appropriate agent
4. **Execution**: Agents execute their tasks sequentially or in parallel
5. **Tracking**: The Tracker monitors progress and handles retries on failure
6. **Output**: Results are compiled into a final formatted output

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific tests
python -m pytest tests/test_planner.py -v
python -m pytest tests/test_dispatcher.py -v

# Run dry-run test
python main.py "Research Python 3.14 new features and write a summary" --dry-run
```

## License

MIT
