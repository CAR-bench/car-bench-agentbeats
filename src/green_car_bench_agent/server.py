"""Server entry point for CAR-bench evaluator agent."""
import argparse
import asyncio
import os
import sys
from pathlib import Path
import warnings

import uvicorn

# Suppress Pydantic serialization warnings from litellm types
# These occur because litellm's Message/Choices types don't set all optional fields
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
    module="pydantic.main"
)

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agentbeats.green_executor import GreenExecutor
from car_bench_evaluator import CARBenchEvaluator

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
sys.path.pop(0)

logger = configure_logger(role="evaluator", context="server")


def car_bench_evaluator_agent_card(name: str, url: str) -> AgentCard:
    """Create the agent card for the CAR-bench evaluator."""
    skill = AgentSkill(
        id="car_bench_evaluation",
        name="CAR-bench Evaluation",
        description="Evaluates agents on CAR-bench voice assistant tasks",
        tags=["benchmark", "evaluation", "car-bench"],
        examples=[
            '{"participants": {"agent": "http://localhost:8080"}, "config": {"num_tasks": 3}}'
        ],
    )
    return AgentCard(
        name=name,
        description="CAR-bench evaluator - tests agents on in-car voice assistant tasks",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


async def main():
    parser = argparse.ArgumentParser(description="Run the CAR-bench evaluator agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    args = parser.parse_args()

    # Auto-configure CAR_BENCH_DATA_DIR if not set
    if "CAR_BENCH_DATA_DIR" not in os.environ:
        # Default to scenarios/car-bench/car-bench/mock_data if it exists
        project_root = Path(__file__).parent.parent.parent
        default_data_dir = project_root / "scenarios" / "car-bench" / "car-bench" / "car_bench" / "envs" / "car_voice_assistant" / "mock_data"
        if default_data_dir.exists():
            os.environ["CAR_BENCH_DATA_DIR"] = str(default_data_dir)
            logger.info(f"Auto-configured CAR_BENCH_DATA_DIR={default_data_dir}")
        else:
            logger.warning(
                f"CAR_BENCH_DATA_DIR not set and default path not found: {default_data_dir}. "
                "Run ./scenarios/car-bench/setup.sh to download data."
            )

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"

    logger.info(
        "Starting CAR-bench evaluator server",
        host=args.host,
        port=args.port,
        url=agent_url
    )

    agent = CARBenchEvaluator()
    executor = GreenExecutor(agent)
    agent_card = car_bench_evaluator_agent_card("CARBenchEvaluator", agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
