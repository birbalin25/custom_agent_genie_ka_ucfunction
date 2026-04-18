import logging
from typing import Any, AsyncGenerator, AsyncIterator, Optional

from databricks.sdk import WorkspaceClient
from databricks_langchain.chat_models import json
from langchain.messages import AIMessage, ToolMessage
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
)


def get_session_id(request: ResponsesAgentRequest) -> str | None:
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        return request.custom_inputs.get("session_id")
    return None


def get_user_workspace_client() -> WorkspaceClient:
    token = get_request_headers().get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")


def get_databricks_host_from_env() -> Optional[str]:
    try:
        w = WorkspaceClient()
        return w.config.host
    except Exception as e:
        logging.exception(f"Error getting databricks host from env: {e}")
        return None


async def process_agent_astream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """
    Generic helper to process agent stream events and yield ResponsesAgentStreamEvent objects.

    Args:
        async_stream: The async iterator from agent.astream()
    """
    async for event in async_stream:
        if event[0] == "updates":
            for node_data in event[1].values():
                messages = node_data.get("messages", [])
                if messages:
                    # Only keep AIMessages without tool_calls (final text responses).
                    # This filters out function_call, function_call_output, and
                    # intermediate "thinking" messages from the UI.
                    filtered = [
                        msg for msg in messages
                        if not isinstance(msg, ToolMessage)
                        and not (isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None))
                    ]
                    if filtered:
                        for item in output_to_responses_items_stream(filtered):
                            yield item
