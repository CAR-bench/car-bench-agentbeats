"""
CAR-bench Agent - Purple agent that solves CAR-bench tasks.

This is the agent being tested. It:
1. Receives task descriptions with available tools from the green agent
2. Decides which tool to call or how to respond
3. Returns responses in the expected JSON format wrapped in <json>...</json> tags
"""
import argparse
import json
import os
from pathlib import Path
import sys
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, Part, TextPart, DataPart, Role
from a2a.utils import new_agent_parts_message
from litellm import completion
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
sys.path.pop(0)

logger = configure_logger(role="agent", context="-")

SYSTEM_PROMPT = """You are a helpful car voice assistant. Follow the policy and tool instructions provided."""


class CARBenchAgentExecutor(AgentExecutor):
    """Executor for the CAR-bench purple agent using native tool calling."""

    def __init__(self, model: str, temperature: float = 0.0, thinking: bool = False, reasoning_effort: str = "medium", interleaved_thinking: bool = False):
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort  # Can be 'none', 'disable', 'low', 'medium', 'high', or integer token budget
        self.interleaved_thinking = interleaved_thinking  # Whether to use interleaved thinking
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        inbound_message = context.message
        ctx_logger = logger.bind(role="agent", context=f"ctx:{context.context_id[:8]}")
        
        # Initialize or get conversation history
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []

        messages = self.ctx_id_to_messages[context.context_id]
        tools = self.ctx_id_to_tools.get(context.context_id, [])

        # Parse the incoming A2A Message with Parts
        user_message_text = None
        
        try:
            for part in inbound_message.parts:
                if isinstance(part.root, TextPart):
                    text = part.root.text
                    # Parse system prompt and user message from formatted text
                    if "System:" in text and "\n\nUser:" in text:
                        # First message with system prompt
                        parts = text.split("\n\nUser:", 1)
                        system_prompt = parts[0].replace("System:", "").strip()
                        user_message_text = parts[1].strip()
                        if not messages:  # Only add system prompt once
                            messages.append({"role": "system", "content": system_prompt})
                    else:
                        # Regular user message
                        user_message_text = text
                
                elif isinstance(part.root, DataPart):
                    # Extract tools from DataPart
                    data = part.root.data
                    if "tools" in data:
                        tools = data["tools"]
                        self.ctx_id_to_tools[context.context_id] = tools
            
            # Fallback if no text part found
            if not user_message_text:
                user_message_text = context.get_user_input()
            
            ctx_logger.info(
                "Received user message",
                context_id=context.context_id[:8],
                turn=len(messages) + 1,
                message_preview=user_message_text[:100] if user_message_text else ""
            )
            ctx_logger.debug(
                "Message details",
                context_id=context.context_id[:8],
                message=user_message_text,
                num_parts=len(inbound_message.parts),
                has_tools=bool(tools),
                num_tools=len(tools) if tools else 0
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse message parts: {e}, using fallback")
            user_message_text = context.get_user_input()

        # Check if previous message had tool calls - if so, format as tool results
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
            # This message contains tool results - format them for Claude
            tool_call_ids = [tc["id"] for tc in messages[-1]["tool_calls"]]
            
            # Parse tool results from the user message
            # CAR-bench sends tool results as text like "Tool: get_location\nResult: {...}"
            # We need to match each result with its tool_call_id
            tool_results = []
            for i, tool_call in enumerate(messages[-1]["tool_calls"]):
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": user_message_text  # For now, send full message to each tool
                })
            
            # Add all tool result messages
            messages.extend(tool_results)
            
            ctx_logger.debug(
                "Formatted tool results",
                num_tools=len(tool_results),
                tool_call_ids=[tr["tool_call_id"] for tr in tool_results]
            )
        else:
            # Regular user message
            messages.append({"role": "user", "content": user_message_text})

        # Call LLM with native tool calling
        try:
            # Configure prompt caching
            tools[-1]["function"]["cache_control"] = {"type": "ephemeral"} # tools
            messages[0]["cache_control"] = {"type": "ephemeral"} # system prompt

            completion_kwargs = {
                "model": self.model,
                "tools": tools if tools else None,
                "temperature": self.temperature,
            }

            # Configure reasoning effort / thinking
            if self.thinking:
                    if self.model == "claude-opus-4-6":
                        completion_kwargs["thinking"] = {
                            "type": "adaptive"
                        }
                    else:
                        if self.reasoning_effort in [
                            "none",
                            "disable",
                            "low",
                            "medium",
                            "high",
                        ]:
                            completion_kwargs["reasoning_effort"] = self.reasoning_effort
                        else:
                            try:
                                thinking_budget = int(self.reasoning_effort)
                            except ValueError:
                                raise ValueError(
                                    "reasoning_effort must be 'none', 'disable', 'low', 'medium', 'high', or an integer value"
                                )
                            completion_kwargs["thinking"] = {
                                "type": "enabled",
                                "budget_tokens": thinking_budget,
                            }
                        if self.interleaved_thinking:
                            completion_kwargs["extra_headers"] = {
                                    "anthropic-beta": "interleaved-thinking-2025-05-14"
                                }


            response = completion(
                messages=messages,
                **completion_kwargs
            )
            
            # Get the message from LLM
            llm_message = response.choices[0].message
            assistant_content = llm_message.model_dump(exclude_unset=True)
            
            # Extract tool calls from assistant content
            tool_calls = assistant_content.get("tool_calls")
            
            ctx_logger.info(
                "LLM response received",
                has_tool_calls=bool(tool_calls),
                num_tool_calls=len(tool_calls) if tool_calls else 0,
                has_content=bool(assistant_content.get("content")),
                content_length=len(assistant_content.get("content") or ""),
                has_thinking=bool(assistant_content.get("thinking_blocks") or assistant_content.get("reasoning_content"))
            )
            ctx_logger.debug(
                "LLM response details",
                context_id=context.context_id[:8],
                content=assistant_content.get("content"),
                tool_calls=[{"name": tc["function"]["name"], "args": tc["function"]["arguments"]} for tc in tool_calls] if tool_calls else None,
                reasoning_content=assistant_content.get("reasoning_content")
            )

            # Build proper A2A Message with Parts
            parts = []
            
            # Add TextPart if there's content
            if assistant_content.get("content"):
                parts.append(Part(root=TextPart(
                    kind="text",
                    text=assistant_content["content"]
                )))
            
            # Add DataPart if there are tool calls
            if assistant_content.get("tool_calls"):
                tool_calls_list = [
                    ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in assistant_content["tool_calls"]
                ]
                tool_calls_data = ToolCallsData(tool_calls=tool_calls_list)
                parts.append(Part(root=DataPart(
                    kind="data",
                    data=tool_calls_data.model_dump()
                )))
            
            # Add reasoning_content as DataPart for debugging (if present)
            if assistant_content.get("reasoning_content"):
                parts.append(Part(root=DataPart(
                    kind="data",
                    data={"reasoning_content": assistant_content["reasoning_content"]}
                )))
            
            # If no parts, add empty text
            if not parts:
                parts.append(Part(root=TextPart(
                    kind="text",
                    text=assistant_content.get("content", "")
                )))
            
            ctx_logger.debug(
                "Sending response",
                context_id=context.context_id[:8],
                num_parts=len(parts),
                parts_summary=[{"kind": p.root.kind, "has_data": bool(p.root.text if hasattr(p.root, 'text') else p.root.data)} for p in parts]
            )
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            # Error response as Parts
            parts = [Part(root=TextPart(
                kind="text",
                text=f"Error processing request: {str(e)}"
            ))]
            # Create a simple assistant_content for error case
            assistant_content = {"content": f"Error processing request: {str(e)}"}

        # Add to history - preserve complete assistant message including thinking blocks
        # Store the full assistant_content to preserve thinking blocks and reasoning_content
        assistant_message_for_history = {
            "role": "assistant",
            "content": assistant_content.get("content"),
        }
        
        # Preserve tool calls in proper format for LLM API
        if assistant_content.get("tool_calls"):
            assistant_message_for_history["tool_calls"] = assistant_content["tool_calls"]
        
        # Preserve thinking blocks and reasoning content for Claude extended thinking
        if assistant_content.get("thinking_blocks"):
            assistant_message_for_history["thinking_blocks"] = assistant_content["thinking_blocks"]
        if assistant_content.get("reasoning_content"):
            assistant_message_for_history["reasoning_content"] = assistant_content["reasoning_content"]
        
        messages.append(assistant_message_for_history)

        # Send response via A2A - use new_agent_parts_message
        response_message = new_agent_parts_message(
            parts=parts,
            context_id=context.context_id
        )
        await event_queue.enqueue_event(response_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution."""
        logger.bind(role="agent", context=f"ctx:{context.context_id[:8]}").info(
            "Canceling context",
            context_id=context.context_id[:8]
        )
        if context.context_id in self.ctx_id_to_messages:
            del self.ctx_id_to_messages[context.context_id]
        if context.context_id in self.ctx_id_to_tools:
            del self.ctx_id_to_tools[context.context_id]
