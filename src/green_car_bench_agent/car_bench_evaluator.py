"""
CAR-bench Evaluator - Green agent that runs CAR-bench evaluation on purple agents.

This agent:
1. Sets up CAR-bench voice assistant environments
2. Sends task prompts to the purple agent being tested 
(wrapped in a RemoteA2AAgent that communicates via A2A protocol)
3. Parses the purple agent's tool-call responses
4. Steps through the environment and collects metrics
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import nest_asyncio
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest
from agentbeats.tool_provider import ToolProvider

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
sys.path.pop(0)

# Import run.py from car-bench repo root
car_bench_repo = Path(__file__).parent.parent.parent / "scenarios" / "car-bench" / "car-bench"
sys.path.insert(0, str(car_bench_repo))
from run import run as run_benchmark
sys.path.pop(0)

# Import from car_bench package
from car_bench.types import Action, EnvRunResult

nest_asyncio.apply()
logger = configure_logger(role="evaluator", context="-")

RESPOND_ACTION_NAME = "respond"


def create_remote_agent_factory(agent_url: str):
    """Create a factory that produces RemoteA2AAgent instances.
    
    Each agent gets its own ToolProvider to avoid threading issues.
    """
    def factory(tools_info, wiki, args):
        # Import Agent base class and types
        from car_bench.agents.base import Agent
        from car_bench.types import AgentState
        
        # Create an agent that delegates to remote purple agent via A2A
        class RemoteA2AAgent(Agent):
            def __init__(self, agent_url: str):
                self.agent_url = agent_url
                self.tool_provider = ToolProvider()
                self._is_first_message = True
            
            def get_init_state(self, system_prompt: str, initial_observation: str) -> AgentState:
                """Initialize agent state with system prompt and initial observation."""
                self._is_first_message = True
                return AgentState(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": initial_observation},
                    ]
                )
            
            def generate_next_message(self, state: AgentState, tools_info: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], AgentState]:
                """Generate next message by calling remote purple agent."""
                import asyncio
                
                # Extract last user message
                last_user_msg = state.messages[-1]["content"]
                
                # Handle empty messages - replace with placeholder to avoid LLM errors
                if not last_user_msg or not last_user_msg.strip():
                    logger.warning(
                        "Empty user message detected, using placeholder 'none'",
                        message_index=len(state.messages) - 1
                    )
                    last_user_msg = "none"
                
                # Build proper A2A message with Parts
                if self._is_first_message:
                    # First message: combine system prompt and user message in one TextPart,
                    # send tools as separate DataPart
                    parts = []
                    
                    # Combine system prompt and user message into single TextPart
                    system_prompt = state.messages[0]["content"] if state.messages[0]["role"] == "system" else ""
                    prompt_text = f"System: {system_prompt}\n\nUser: {last_user_msg}" if system_prompt else last_user_msg
                    parts.append(Part(root=TextPart(
                        kind="text",
                        text=prompt_text
                    )))
                    
                    # Add tools as DataPart (structured data)
                    if tools_info:
                        parts.append(Part(root=DataPart(
                            kind="data",
                            data={"tools": tools_info}
                        )))
                else:
                    # Subsequent messages: just the user message
                    parts = [Part(root=TextPart(
                        kind="text",
                        text=last_user_msg
                    ))]
                
                # Call remote agent via A2A
                # Use synchronous call since we're in a thread pool executor
                is_new_conversation = self._is_first_message
                self._is_first_message = False
                
                logger.debug(
                    "Sending message to purple agent",
                    agent_url=self.agent_url,
                    new_conversation=is_new_conversation,
                    num_parts=len(parts),
                    parts_summary=[{"kind": p.root.kind} for p in parts]
                )
                
                # Use synchronous method to avoid event loop issues in thread pool
                response = self.tool_provider.talk_to_agent_with_parts_sync(
                    parts=parts,
                    url=self.agent_url,
                    new_conversation=is_new_conversation,
                )
                
                logger.debug(
                    "Received response from purple agent",
                    agent_url=self.agent_url,
                    response_type=type(response).__name__,
                    has_parts=hasattr(response, 'parts')
                )
                
                # Parse response into standard message format
                next_message = self._parse_response(response)
                
                # Update state
                updated_state = AgentState(
                    messages=state.messages + [next_message],
                    total_cost=state.total_cost,
                    total_llm_induced_latency_ms=state.total_llm_induced_latency_ms,
                    turn_counter=state.turn_counter,
                    least_prompt_tokens=state.least_prompt_tokens,
                    latest_prompt_tokens=state.latest_prompt_tokens,
                )
                
                return next_message, updated_state
            
            def _parse_response(self, response) -> Dict[str, Any]:
                """Parse the A2A Message response into standard agent message format."""
                try:
                    # Response is now a Message object with parts
                    from a2a.types import Message
                    
                    if not isinstance(response, Message):
                        # Fallback: try parsing as JSON string
                        parsed = json.loads(response)
                        parts = parsed.get("parts", [])
                    else:
                        # Direct Message object - use parts directly
                        parts = response.parts
                    
                    content = None
                    tool_calls = None
                    reasoning_content = None
                    
                    # Process each part
                    for part in parts:
                        # Handle both Pydantic Part objects and dict representations
                        if hasattr(part, 'root'):
                            # Pydantic Part object
                            from a2a.types import TextPart, DataPart
                            if isinstance(part.root, TextPart):
                                content = part.root.text
                            elif isinstance(part.root, DataPart):
                                data = part.root.data
                                if "tool_calls" in data:
                                    tool_calls = [
                                        {
                                            "id": f"call_{hash(json.dumps(tc)) % 100000000:08x}",
                                            "type": "function",
                                            "function": {
                                                "name": tc["tool_name"],
                                                "arguments": json.dumps(tc["arguments"]),
                                            },
                                        }
                                        for tc in data["tool_calls"]
                                    ]
                                elif "reasoning_content" in data:
                                    reasoning_content = data["reasoning_content"]
                        else:
                            # Dict representation
                            part_kind = part.get("root", {}).get("kind") or part.get("kind")
                            
                            if part_kind == "text":
                                text = part.get("root", {}).get("text") or part.get("text")
                                if text:
                                    content = text
                            
                            elif part_kind == "data":
                                data = part.get("root", {}).get("data") or part.get("data")
                                if data and "tool_calls" in data:
                                    tool_calls = [
                                        {
                                            "id": f"call_{hash(json.dumps(tc)) % 100000000:08x}",
                                            "type": "function",
                                            "function": {
                                                "name": tc["tool_name"],
                                                "arguments": json.dumps(tc["arguments"]),
                                            },
                                        }
                                        for tc in data["tool_calls"]
                                    ]
                                elif data and "reasoning_content" in data:
                                    reasoning_content = data["reasoning_content"]
                    
                    parsed_msg = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                    
                    # Include reasoning_content for debugging if present
                    if reasoning_content:
                        parsed_msg["reasoning_content"] = reasoning_content
                    
                    logger.debug(
                        "Parsed purple agent response",
                        has_content=bool(content),
                        content_preview=content[:100] if content else None,
                        has_tool_calls=bool(tool_calls),
                        tool_calls=tool_calls,
                        has_reasoning=bool(reasoning_content),
                        reasoning_preview=reasoning_content[:100] if reasoning_content else None
                    )
                    
                    return parsed_msg
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse agent response: {e}")
                    # If parsing fails, treat as plain text response
                    return {
                        "role": "assistant",
                        "content": response,
                        "tool_calls": None,
                    }
        
        return RemoteA2AAgent(agent_url=agent_url)
    
    return factory


def calculate_evaluation_results(
    results_by_split: Dict[str, List[EnvRunResult]],
    time_used: float
) -> Tuple[Dict[str, Any], str]:
    """Calculate comprehensive evaluation results and format summary.
    
    Args:
        results_by_split: Results organized by task split (base, hallucination, disambiguation)
        time_used: Total evaluation time in seconds
        
    Returns:
        Tuple of (result_data dict, summary string)
    """
    # Import analysis functions from car-bench repo root
    car_bench_repo = Path(__file__).parent.parent.parent / "scenarios" / "car-bench" / "car-bench"
    sys.path.insert(0, str(car_bench_repo))
    try:
        from analyze_results_v2 import (
            organize_data_by_task_and_trial,
            calculate_pass_power_k_scores,
            calculate_pass_at_k_scores,
        )
    finally:
        sys.path.pop(0)
    
    # Flatten all results
    all_results = [r for results in results_by_split.values() for r in results]
    total_reward = sum(r.reward for r in all_results)
    num_completed = len(all_results)
    pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0
    
    # Split task rewards by task type
    task_rewards_by_split = {
        split: {str(r.task_id): r.reward for r in results}
        for split, results in results_by_split.items()
        if results
    }
    
    # Calculate metrics for each split separately
    pass_power_k_scores_by_split = {}
    pass_at_k_scores_by_split = {}
    max_trials = 1
    
    for split, results in results_by_split.items():
        if not results:
            continue
            
        # Convert results to format expected by analyze_results.py
        analysis_data = [
            {
                "task_id": result.task_id,
                "reward": result.reward,
                "info": result.info,
                "trial": result.trial,
            }
            for result in results
        ]
        
        # Organize data and calculate metrics for this split
        organized_data = organize_data_by_task_and_trial(analysis_data)
        split_max_trials = (
            max(len(trials) for trials in organized_data.values())
            if organized_data else 1
        )
        max_trials = max(max_trials, split_max_trials)
        
        pass_power_k_scores_by_split[split] = calculate_pass_power_k_scores(organized_data, split_max_trials)
        pass_at_k_scores_by_split[split] = calculate_pass_at_k_scores(organized_data, split_max_trials)
    
    # Calculate overall metrics as average across splits
    pass_power_k_scores, pass_at_k_scores = calculate_average_metrics_across_splits(
        pass_power_k_scores_by_split,
        pass_at_k_scores_by_split,
        max_trials
    )
    
    # Prepare detailed results with reward_info, task info, and trajectories - split by task type
    detailed_results_by_split = {}
    
    for split, results in results_by_split.items():
        if not results:
            continue
            
        detailed_results_by_split[split] = [
            {
                "task_id": result.task_id,
                "reward": result.reward,
                "trial": result.trial,
                "reward_info": result.info.get("reward_info", {}),
                "task": result.info.get("task", {}),
                "trajectory": [
                    msg for msg in result.traj
                    if msg.get("role") != "system"
                ],
                "user_cost": result.info.get("user_cost", 0),
                "total_agent_cost": result.info.get("total_agent_cost", 0),
                "total_llm_latency_ms": result.info.get("total_llm_induced_latency_ms", 0),
            }
            for result in results
        ]
    
    # Format task results for display by split
    task_results_by_split_str = []
    for split in ["base", "hallucination", "disambiguation"]:
        if split in results_by_split and results_by_split[split]:
            results = results_by_split[split]
            split_results = "\n".join(
                f"    Task {r.task_id}: {'✓' if r.reward >= 0.99 else '✗'} ({r.reward:.2f})"
                for r in results
            )
            split_reward = sum(r.reward for r in results)
            split_count = len(results)
            split_pass_rate = (split_reward / split_count * 100) if split_count > 0 else 0
            task_results_by_split_str.append(
                f"  {split.capitalize()}: {split_pass_rate:.1f}% ({split_reward:.1f}/{split_count})\n{split_results}"
            )
    
    task_results_str = "\n\n".join(task_results_by_split_str)
    
    # Format Pass^k and Pass@k scores
    pass_scores_str = [
        f"  Pass^{k}: {pass_power_k_scores.get(f'Pass^{k}', 0) * 100:.1f}%  |  Pass@{k}: {pass_at_k_scores.get(f'Pass@{k}', 0) * 100:.1f}%"
        for k in range(1, max_trials + 1)
    ]
    pass_scores_display = "\n".join(pass_scores_str)
    
    # Build result data
    result_data = {
        "score": total_reward,
        "max_score": num_completed,
        "pass_rate": pass_rate,
        "task_rewards_by_split": task_rewards_by_split,
        "time_used": time_used,
        "pass_power_k_scores": pass_power_k_scores,
        "pass_at_k_scores": pass_at_k_scores,
        "pass_power_k_scores_by_split": pass_power_k_scores_by_split,
        "pass_at_k_scores_by_split": pass_at_k_scores_by_split,
        "max_trials": max_trials,
        "detailed_results_by_split": detailed_results_by_split,
    }
    
    # Build summary string
    summary = f"""CAR-bench Results
Tasks: {num_completed}
Overall Pass Rate: {pass_rate:.1f}% ({total_reward:.1f}/{num_completed})
Time: {time_used:.1f}s

Pass Scores:
{pass_scores_display}

Task Results by Split:
{task_results_str}"""
    
    return result_data, summary


def calculate_average_metrics_across_splits(
    pass_power_k_scores_by_split: Dict[str, Dict[str, float]],
    pass_at_k_scores_by_split: Dict[str, Dict[str, float]],
    max_trials: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Calculate average metrics across splits (not weighted by task count).
    
    Returns:
        Tuple of (pass_power_k_scores, pass_at_k_scores)
    """
    num_splits = len(pass_power_k_scores_by_split)
    if num_splits == 0:
        return {}, {}
    
    # Average Pass^k and Pass@k scores across splits
    pass_power_k_scores = {}
    pass_at_k_scores = {}
    
    for k in range(1, max_trials + 1):
        pass_power_key = f"Pass^{k}"
        pass_at_key = f"Pass@{k}"
        
        # Sum scores across splits
        pass_power_sum = sum(
            scores.get(pass_power_key, 0.0)
            for scores in pass_power_k_scores_by_split.values()
            if pass_power_key in scores
        )
        pass_at_sum = sum(
            scores.get(pass_at_key, 0.0)
            for scores in pass_at_k_scores_by_split.values()
            if pass_at_key in scores
        )
        
        pass_power_k_scores[pass_power_key] = pass_power_sum / num_splits
        pass_at_k_scores[pass_at_key] = pass_at_sum / num_splits
    
    return pass_power_k_scores, pass_at_k_scores


def build_args_from_config(config: dict, task_type: str) -> argparse.Namespace:
    """Convert evaluation config to run() arguments for a specific task type."""
    return argparse.Namespace(
        env="car_voice_assistant",
        task_type=task_type,
        task_split=config.get("task_split", "test"),
        num_tasks=config.get(f"tasks_{task_type}_num_tasks", -1),
        task_id_filter=config.get(f"tasks_{task_type}_task_id_filter", None),
        num_trials=config.get("num_trials", 1),
        max_concurrency=1,  # Sequential to avoid overloading purple agent
        # User simulator settings
        user_strategy="llm",
        user_model=config.get("user_model", "gemini/gemini-2.5-flash"),
        user_model_provider=config.get("user_provider", "gemini"),
        user_thinking=config.get("user_thinking", True),
        # Policy evaluator settings
        policy_evaluator_strategy="llm",
        policy_evaluator_model=config.get("policy_evaluator_model", "gemini/gemini-2.5-flash"),
        policy_evaluator_model_provider=config.get("policy_evaluator_provider", "gemini"),
        evaluate_policy=True,
        score_tool_execution_errors=True,
        score_policy_errors=True,
        # Agent settings (NOT USED for custom agent factory, but required by some code paths)
        agent_strategy="tool-calling",  # Default strategy if factory not used
        model="remote-agent",  # Placeholder, not used for remote agents
        model_provider="a2a", # not used
        temperature=0.0, # not used
        thinking=False, # not used
        interleaved_thinking=False, # not used
        reasoning_effort="none", # not used
        # =======
        use_user_as_a_tool_tools=False,
        planning_and_thinking_tool=True,
        remove_non_standard_fields_from_tools=False,
        few_shot_displays_path=None,
        seed=10,
        shuffle=False,
    )


class CARBenchEvaluator(GreenAgent):
    """Green agent that evaluates purple agents using CAR-bench."""

    def __init__(self):
        self._required_roles = ["agent"]  # The purple agent being tested
        self._required_config_keys = []
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        eval_logger = logger.bind(role="evaluator", context="eval")
        eval_logger.info(
            "Starting CAR-bench evaluation",
            agent_url=str(req.participants["agent"]),
            num_trials=req.config.get("num_trials", 1)
        )
        start_time = time.time()

        # Get the purple agent URL
        agent_url = str(req.participants["agent"])
        
        # Create agent factory
        agent_factory = create_remote_agent_factory(agent_url)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of CAR-bench tasks")
        )

        all_results: List[EnvRunResult] = []
        results_by_split: Dict[str, List[EnvRunResult]] = {
            "base": [],
            "hallucination": [],
            "disambiguation": []
        }
        
        try:
            # Run each task type (base, hallucination, disambiguation)
            for task_type in ["base", "hallucination", "disambiguation"]:
                num_tasks_key = f"tasks_{task_type}_num_tasks"
                task_id_filter_key = f"tasks_{task_type}_task_id_filter"
                
                # Skip if not configured
                if num_tasks_key not in req.config and task_id_filter_key not in req.config:
                    eval_logger.info(
                        "Skipping task type (not configured)",
                        task_type=task_type
                    )
                    continue
                
                split_logger = logger.bind(role="evaluator", context=f"type:{task_type}")
                
                # Build args for this task type
                args = build_args_from_config(req.config, task_type)
                
                # Log task configuration
                task_desc = f"{task_type} tasks (split={args.task_split}"
                if args.task_id_filter:
                    task_desc += f", ids={args.task_id_filter}"
                elif args.num_tasks > 0:
                    task_desc += f", first {args.num_tasks} tasks"
                else:
                    task_desc += ", all tasks"
                task_desc += ")"
                
                split_logger.info(
                    "Starting task type evaluation",
                    task_type=task_type,
                    task_split=args.task_split,
                    num_tasks=args.num_tasks,
                    task_id_filter=args.task_id_filter,
                    num_trials=req.config.get("num_trials", 1)
                )
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Starting evaluation: {task_desc}"
                    )
                )
                
                # Build checkpoint path
                ckpt_path = f"/tmp/car_bench_eval_{task_type}_{args.task_split}.json"
                
                # Clean up any existing checkpoint file to avoid JSON parse errors
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    eval_logger.debug("Removed existing checkpoint file", path=ckpt_path)
                
                # Run in executor to avoid blocking async event loop
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    run_benchmark,
                    args,
                    ckpt_path,
                    agent_factory
                )
                
                all_results.extend(results)
                results_by_split[task_type].extend(results)
                
                # Log completion with summary stats
                split_reward = sum(r.reward for r in results)
                split_logger.info(
                    "Completed task type",
                    task_type=task_type,
                    num_tasks=len(results),
                    total_reward=split_reward,
                    pass_rate=f"{(split_reward / len(results) * 100) if results else 0:.1f}%"
                )

            # Calculate metrics and format results
            time_used = time.time() - start_time
            result_data, summary = calculate_evaluation_results(results_by_split, time_used)

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result_data)),
                ],
                name="Result",
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Evaluation failed: {e}")
            )
            raise
