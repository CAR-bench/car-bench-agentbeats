"""Server entry point for CAR-bench purple agent."""
import argparse
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

from car_bench_agent import CARBenchAgentExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
sys.path.pop(0)

logger = configure_logger(role="agent", context="server")


def prepare_agent_card(url: str) -> AgentCard:
    """Create the agent card for the CAR-bench purple agent."""
    skill = AgentSkill(
        id="car_assistant",
        name="In-Car Voice Assistant",
        description="Helps drivers with navigation, communication, charging, and other in-car tasks",
        tags=["benchmark", "car-bench", "voice-assistant"],
        examples=[],
    )
    return AgentCard(
        name="car_bench_agent",
        description="In-car voice assistant agent for CAR-bench evaluation",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


def main():
    parser = argparse.ArgumentParser(description="Run the CAR-bench agent (purple agent).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    parser.add_argument(
        "--agent-llm", 
        type=str, 
        default=None,  # Will use env var or fallback
        help="LLM model (can also be set via AGENT_LLM env var)"
    )
    args = parser.parse_args()
    
    # Support both command-line args and environment variables
    # Priority: CLI args > env vars > default
    import os
    agent_llm = args.agent_llm or os.getenv("AGENT_LLM", "gemini/gemini-2.5-flash")

    logger.info(
        "Starting CAR-bench agent",
        model=agent_llm,
        host=args.host,
        port=args.port
    )
    
    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=CARBenchAgentExecutor(model=agent_llm),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=1000,
    )


if __name__ == "__main__":
    main()
