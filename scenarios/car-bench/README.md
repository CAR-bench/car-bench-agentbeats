# CAR-bench Scenario

This scenario evaluates agents on the [CAR-bench](https://github.com/CAR-bench/car-bench) (Car-Assistant Recognition Benchmark), which tests AI agents on 101 in-car voice assistant tasks.

## Setup

### 1. Install dependencies

```bash
uv sync --extra car-bench-agent --extra car-bench-evaluator
```

This installs:
- **car-bench-agent** extras: LLM dependencies for the purple agent (google-adk, google-genai, litellm)
- **car-bench-evaluator** extras: car-bench package for the green evaluator (tasks and mock data are automatically loaded from HuggingFace)

### 2. Set your API keys

Create a `.env` file with your API keys:

```bash
cp sample.env .env
```

Edit `.env` and add:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

## Running the Benchmark

### Local Python Execution

```bash
uv run agentbeats-run scenarios/scenario.toml
```

### Docker Compose (for production-like testing)

```bash
# Generate docker-compose.yml
python generate_compose.py --scenario scenarios/scenario-docker-local.toml

# Run
mkdir -p output
docker compose up --abort-on-container-exit

# View results
cat output/results.json
```

## Configuration

The benchmark is configured via `scenarios/scenario.toml` and `scenarios/scenario-docker-local.toml`:

- **num_trials**: Number of times to run each task (for pass@k metrics)
- **task_split**: Which dataset split to use (`"train"` or `"test"`)
- **tasks_base_num_tasks**: Number of base tasks to run (first N tasks, -1 for all)
- **tasks_hallucination_num_tasks**: Number of hallucination tasks to run
- **tasks_disambiguation_num_tasks**: Number of disambiguation tasks to run
- **tasks_*_task_id_filter**: Alternative to num_tasks - specify exact task IDs (e.g., `["base_0", "base_2"]`)
- **max_steps**: Maximum conversation turns per task

## Troubleshooting

**Import errors**: Make sure you installed the extras: `uv sync --extra car-bench-agent --extra car-bench-evaluator`
