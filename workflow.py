from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model


# =========================
# 1) Dummy business logic
# =========================

def calc_func1(question: str) -> str:
    """
    Example calculation path 1.
    Replace with your real calculation logic.
    """
    # Example: count digits in the question
    digit_count = sum(ch.isdigit() for ch in question)
    return f"calc_func1 result: found {digit_count} digit(s) in the question."


def calc_func2(question: str) -> str:
    """
    Example calculation path 2.
    Replace with your real calculation logic.
    """
    # Example: count words in the question
    word_count = len(question.split())
    return f"calc_func2 result: found {word_count} word(s) in the question."


# =========================
# 2) Very small RAG stub
# =========================

@dataclass
class Document:
    doc_id: str
    text: str


class SimpleRAG:
    """
    A minimal retrieval component.
    This is intentionally simple:
    - no embeddings
    - no vector DB
    - just keyword overlap scoring
    """
    def __init__(self, documents: List[Document]) -> None:
        self.documents = documents

    def retrieve(self, question: str, top_k: int = 2) -> List[Document]:
        question_terms = set(question.lower().split())

        scored: List[tuple[int, Document]] = []
        for doc in self.documents:
            doc_terms = set(doc.text.lower().split())
            overlap = len(question_terms.intersection(doc_terms))
            scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]


# =========================
# 3) Shared graph state
# =========================

class AgentState(TypedDict):
    question: str
    route: Literal["calc_func1", "calc_func2"] | None
    calculation_result: str
    retrieved_context: str
    draft_answer: str
    critique: str
    revision_count: int
    max_revisions: int
    final_answer: str


# =========================
# 4) Models
# =========================
# LangChain docs currently show init_chat_model as a standard way to initialize
# chat models for these workflows. You can switch provider/model as needed. :contentReference[oaicite:2]{index=2}

generator_llm = init_chat_model("gpt-4o-mini", temperature=0)
critic_llm = init_chat_model("gpt-4o-mini", temperature=0)
reviser_llm = init_chat_model("gpt-4o-mini", temperature=0)


# =========================
# 5) RAG data
# =========================

rag = SimpleRAG(
    documents=[
        Document(
            doc_id="doc_1",
            text="LangGraph is useful for stateful workflows, branching, and loops."
        ),
        Document(
            doc_id="doc_2",
            text="Retrieval augmented generation combines retrieved context with model reasoning."
        ),
        Document(
            doc_id="doc_3",
            text="Critic and reviser loops can improve answer quality in iterative pipelines."
        ),
    ]
)


# =========================
# 6) Graph nodes
# =========================

def route_question(state: AgentState) -> AgentState:
    """
    Decide which calculation function to use.

    Keep it explicit and deterministic here.
    You can replace this with an LLM-based router if desired.
    """
    q = state["question"].lower()

    # Example routing rule:
    # If the question mentions number/math/count => use calc_func1
    # Otherwise use calc_func2
    if any(token in q for token in ["number", "math", "count", "digit", "sum"]):
        route: Literal["calc_func1", "calc_func2"] = "calc_func1"
    else:
        route = "calc_func2"

    return {"route": route}


def do_calc_func1(state: AgentState) -> AgentState:
    result = calc_func1(state["question"])
    return {"calculation_result": result}


def do_calc_func2(state: AgentState) -> AgentState:
    result = calc_func2(state["question"])
    return {"calculation_result": result}


def retrieve_context(state: AgentState) -> AgentState:
    docs = rag.retrieve(state["question"], top_k=2)

    if not docs:
        context = "No relevant context was retrieved."
    else:
        context = "\n\n".join(
            f"[{doc.doc_id}] {doc.text}"
            for doc in docs
        )

    return {"retrieved_context": context}


def generate_answer(state: AgentState) -> AgentState:
    prompt = f"""
You are a helpful assistant.

User question:
{state["question"]}

Calculation result:
{state["calculation_result"]}

Retrieved context:
{state["retrieved_context"]}

Write a clear answer.
Use the retrieved context only when relevant.
If context is weak, say so briefly rather than inventing details.
""".strip()

    response = generator_llm.invoke(prompt)
    return {"draft_answer": response.content}


def critic_node(state: AgentState) -> AgentState:
    prompt = f"""
You are a strict reviewer.

Question:
{state["question"]}

Answer draft:
{state["draft_answer"]}

Retrieved context:
{state["retrieved_context"]}

Review the draft for:
1. correctness
2. relevance to the question
3. correct use of calculation result
4. unsupported claims
5. clarity

Return exactly in this format:

VERDICT: PASS or FAIL
COMMENTS:
- ...
- ...
""".strip()

    response = critic_llm.invoke(prompt)
    return {"critique": response.content}


def revise_node(state: AgentState) -> AgentState:
    prompt = f"""
You are revising an answer using reviewer feedback.

Question:
{state["question"]}

Current draft:
{state["draft_answer"]}

Calculation result:
{state["calculation_result"]}

Retrieved context:
{state["retrieved_context"]}

Reviewer comments:
{state["critique"]}

Produce a better answer.
Fix factual issues, unsupported claims, and clarity problems.
""".strip()

    response = reviser_llm.invoke(prompt)

    return {
        "draft_answer": response.content,
        "revision_count": state["revision_count"] + 1,
    }


def finalize_node(state: AgentState) -> AgentState:
    return {"final_answer": state["draft_answer"]}


# =========================
# 7) Conditional routing
# =========================

def route_after_router(state: AgentState) -> str:
    """
    Decide which calc node to execute.
    """
    return state["route"] or "calc_func2"


def route_after_critic(state: AgentState) -> str:
    """
    Continue revising until:
    - critic passes, or
    - max revisions reached
    """
    critique = state["critique"].upper()
    passed = "VERDICT: PASS" in critique
    reached_limit = state["revision_count"] >= state["max_revisions"]

    if passed or reached_limit:
        return "finalize"

    return "revise"


# =========================
# 8) Build graph
# =========================

builder = StateGraph(AgentState)

builder.add_node("route_question", route_question)
builder.add_node("calc_func1", do_calc_func1)
builder.add_node("calc_func2", do_calc_func2)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_answer", generate_answer)
builder.add_node("critic", critic_node)
builder.add_node("revise", revise_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "route_question")

builder.add_conditional_edges(
    "route_question",
    route_after_router,
    {
        "calc_func1": "calc_func1",
        "calc_func2": "calc_func2",
    },
)

builder.add_edge("calc_func1", "retrieve_context")
builder.add_edge("calc_func2", "retrieve_context")
builder.add_edge("retrieve_context", "generate_answer")
builder.add_edge("generate_answer", "critic")

builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "revise": "revise",
        "finalize": "finalize",
    },
)

builder.add_edge("revise", "critic")
builder.add_edge("finalize", END)

graph = builder.compile()


# =========================
# 9) Example run
# =========================

if __name__ == "__main__":
    initial_state: AgentState = {
        "question": "Can you explain how LangGraph workflows use loops and also count the words in my question?",
        "route": None,
        "calculation_result": "",
        "retrieved_context": "",
        "draft_answer": "",
        "critique": "",
        "revision_count": 0,
        "max_revisions": 2,
        "final_answer": "",
    }

    result = graph.invoke(initial_state)
    print("\nFINAL ANSWER:\n")
    print(result["final_answer"])