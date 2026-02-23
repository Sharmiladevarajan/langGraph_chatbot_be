"""Chat API: LangGraph workflow with conversation memory."""
import uuid
from fastapi import APIRouter
from langchain_core.messages import HumanMessage, AIMessage

from app.models.schemas import ChatRequest, ChatResponse
from app.graph.workflow import build_chat_graph

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory conversation store (session_id -> list of messages)
# In production use Redis or a DB
_conversations: dict[str, list] = {}


def _get_or_create_messages(session_id: str | None):
    sid = session_id or str(uuid.uuid4())
    if sid not in _conversations:
        _conversations[sid] = []
    return sid, _conversations[sid]


async def _chat_handler(request: ChatRequest) -> ChatResponse:
    """Send a message; run LangGraph workflow; return reply and maintain history."""
    session_id, messages = _get_or_create_messages(request.session_id)
    graph = build_chat_graph()

    initial_state = {
        "messages": list(messages),
        "question": request.message,
        "context": "",
        "use_documents": request.use_documents,
        "subject_filter": request.subject_filter,
        "session_id": session_id,
    }
    result = graph.invoke(initial_state)
    new_messages = result.get("messages", [])
    _conversations[session_id] = new_messages

    last_msg = new_messages[-1] if new_messages else None
    reply = last_msg.content if last_msg and hasattr(last_msg, "content") else ""

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        sources=None,
    )


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """POST /chat/ – main endpoint."""
    return await _chat_handler(request)


@router.post("", response_model=ChatResponse, include_in_schema=False)
async def chat_no_slash(request: ChatRequest) -> ChatResponse:
    """POST /chat (no trailing slash) – avoids 307 redirect when client omits slash."""
    return await _chat_handler(request)
