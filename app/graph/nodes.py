"""
LangGraph nodes: retrieve, route, and generate.
- Router: decide if question needs documents or is general chat.
- Retriever: fetch relevant chunks (with optional subject filter).
- Generator: LLM answer using context + conversation history.
"""
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.llm_factory import get_llm
from app.services.vector_store import similarity_search


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], "conversation history"]
    question: str
    context: str
    use_documents: bool
    subject_filter: str | None
    session_id: str | None


def route_question(state: GraphState) -> Literal["retrieve", "generate"]:
    """
    Conditional routing: if use_documents and question looks like it needs docs,
    go to retrieve; else go straight to generate (general chat).
    """
    if not state.get("use_documents", True):
        return "generate"
    q = (state.get("question") or "").strip().lower()
    # Simple heuristic: short greetings -> general; else retrieve
    if q in ("hi", "hello", "hey", "thanks", "thank you", "bye"):
        return "generate"
    return "retrieve"


def retrieve_node(state: GraphState) -> dict:
    """Fetch relevant document chunks; store in state as context."""
    question = state["question"]
    subject = state.get("subject_filter")
    docs = similarity_search(question, k=4, subject_filter=subject)
    context = "\n\n".join(
        f"[{d['metadata'].get('filename', '')}]: {d['content']}" for d in docs
    )
    return {"context": context or "No relevant documents found."}


def generate_node(state: GraphState) -> dict:
    """LLM generates reply using conversation history and (if any) context."""
    messages = state["messages"]
    context = state.get("context") or ""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based on the context below when relevant. "
         "If the context is empty or not relevant, answer from general knowledge.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    llm = get_llm()
    chain = prompt | llm
    history = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    response = chain.invoke({
        "context": context,
        "history": history,
        "question": state["question"],
    })
    new_messages = messages + [HumanMessage(content=state["question"]), response]
    return {"messages": new_messages}


def generate_without_retrieve_node(state: GraphState) -> dict:
    """Generate reply without document context (general chat)."""
    return generate_node({**state, "context": ""})
