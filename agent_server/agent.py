import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

import litellm
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.tools import tool
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server.utils import (
    get_databricks_host_from_env,
    get_session_id,
    get_user_workspace_client,
    process_agent_astream_events,
)

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True
sp_workspace_client = WorkspaceClient()


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


@tool
def knowledge_assistant(query: str) -> str:
    """Search the knowledge base for answers about product specifications, return/refund policy, warranty terms, shipping guidelines, customer FAQ, troubleshooting guides, and membership program information for TechCommerce Inc."""
    response = sp_workspace_client.api_client.do(
        "POST",
        "/serving-endpoints/ka-b3b5b187-endpoint/invocations",
        body={"input": [{"role": "user", "content": query}]},
    )
    # KA endpoints return Responses API format: output[].content[].text
    output = response.get("output", [])
    for item in output:
        if item.get("type") == "message" and item.get("content"):
            return "".join(part.get("text", "") for part in item["content"])
    return "No answer found."


# Placeholder for Knowledge Assistant 2
# @tool
# def knowledge_assistant_2(query: str) -> str:
#     """Search the knowledge base for answers about <DESCRIBE_YOUR_KB_TOPICS>."""
#     response = sp_workspace_client.api_client.do(
#         "POST",
#         "/serving-endpoints/<REPLACE_KA_ENDPOINT_NAME>/invocations",
#         body={"input": [{"role": "user", "content": query}]},
#     )
#     output = response.get("output", [])
#     for item in output:
#         if item.get("type") == "message" and item.get("content"):
#             return "".join(part.get("text", "") for part in item["content"])
#     return "No answer found."


def init_mcp_client(workspace_client: WorkspaceClient) -> DatabricksMultiServerMCPClient:
    host_name = get_databricks_host_from_env()
    return DatabricksMultiServerMCPClient(
        [
            # Genie Space 1
            DatabricksMCPServer(
                name="genie-space-1",
                url=f"{host_name}/api/2.0/mcp/genie/01f134fb5f9b1dd78966fd2bf47f26a2",
                workspace_client=workspace_client,
            ),
            # Placeholder for Genie Space 2
            # DatabricksMCPServer(
            #     name="genie-space-2",
            #     url=f"{host_name}/api/2.0/mcp/genie/<REPLACE_GENIE_SPACE_2_ID>",
            #     workspace_client=workspace_client,
            # ),
            # placeholder for vector search index
            # DatabricksMCPServer(
            #     name="my-vector-search",
            #     url=f"{host_name}/api/2.0/mcp/vector-search/<CATALOG>/<SCHEMA>/<INDEX_NAME>",
            #     workspace_client=workspace_client,
            # ),

            # UC Function: get_city_temperature
            DatabricksMCPServer(
                name="get-city-temperature",
                url=f"{host_name}/api/2.0/mcp/functions/serverless_stable_14ey07_catalog/mc/get_city_temperature",
                workspace_client=workspace_client,
            ),
        ]
    )


async def init_agent(workspace_client: Optional[WorkspaceClient] = None):
    tools = [get_current_time, knowledge_assistant]  # Add knowledge_assistant_2 here when ready
    mcp_client = init_mcp_client(workspace_client or sp_workspace_client)
    try:
        tools.extend(await mcp_client.get_tools())
    except Exception:
        logger.warning("Failed to fetch MCP tools. Continuing without MCP tools.", exc_info=True)
    return create_agent(tools=tools, model=_SanitizedChatDatabricks(endpoint="databricks-claude-sonnet-4-6"))


class _SanitizedChatDatabricks(ChatDatabricks):
    """Strips extra fields (e.g. 'id') from tool message content blocks
    that some LLM APIs reject."""

    @staticmethod
    def _strip_content_ids(messages):
        for msg in messages:
            if isinstance(msg.content, list):
                msg.content = [
                    {k: v for k, v in block.items() if k != "id"}
                    if isinstance(block, dict) else block
                    for block in msg.content
                ]
        return messages

    def _stream(self, messages, *args, **kwargs):
        return super()._stream(self._strip_content_ids(messages), *args, **kwargs)

    async def _astream(self, messages, *args, **kwargs):
        async for chunk in super()._astream(self._strip_content_ids(messages), *args, **kwargs):
            yield chunk


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    # By default, uses service principal credentials.
    # For on-behalf-of user authentication, use get_user_workspace_client() instead:
    #   agent = await init_agent(workspace_client=get_user_workspace_client())
    agent = await init_agent()
    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}

    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event
