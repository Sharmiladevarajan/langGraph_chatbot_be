"""
LangGraph workflow: route -> retrieve (optional) -> generate.
Conversation history is maintained in state and passed each turn.
"""
from langgraph.graph import StateGraph, END
from app.graph.nodes import GraphState, route_question, retrieve_node, generate_node


def build_chat_graph():
    """Build the graph: conditional routing then retrieve + generate or generate only."""
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_conditional_edges("__start__", route_question, {
        "retrieve": "retrieve",
        "generate": "generate",
    })
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
