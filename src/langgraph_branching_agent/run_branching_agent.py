from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# Define state structure
class State(TypedDict):
    input: str
    result: str


# Define our possible nodes (tasks)
def research_node(state: State) -> State:
    print("Research node: Gathering information...")
    state["result"] = "Here is the research report."
    return state


def analysis_node(state: State) -> State:
    print("Analysis node: Analyzing data...")
    state["result"] = "Here is the analysis summary."
    return state


def escalation_node(state: State) -> State:
    print("Escalation node: Escalating to human expert...")
    state["result"] = "Escalated to human expert for further review."
    return state


# Decision function: routes input to the right node
def decision_router(state: State) -> str:
    input_text = state["input"]
    print(f"Routing based on input: '{input_text}'")
    # Simple rules for demonstration
    if "urgent" in input_text.lower() or "error" in input_text.lower():
        return "escalation"
    elif len(input_text.split()) > 12:
        return "analysis"
    else:
        return "research"


# Define the workflow graph
graph = StateGraph(State)

# Add nodes
graph.add_node("research", research_node)
graph.add_node("analysis", analysis_node)
graph.add_node("escalation", escalation_node)

# Add conditional edges from START
graph.add_conditional_edges(
    START,
    decision_router,
    {"research": "research", "analysis": "analysis", "escalation": "escalation"},
)

# Connect all nodes to END
graph.add_edge("research", END)
graph.add_edge("analysis", END)
graph.add_edge("escalation", END)

# Compile the graph
workflow = graph.compile()

# Visualize the graph
graph_representation = workflow.get_graph()

# Save Mermaid diagram to file
mermaid_diagram = graph_representation.draw_mermaid()
with open("workflow_graph.mmd", "w") as f:
    f.write(mermaid_diagram)
print("Mermaid diagram saved to workflow_graph.mmd")


# Example runs
if __name__ == "__main__":
    test_inputs = [
        {
            "input": "Summarize the quarterly sales data for the last year.",
            "result": "",
        },
        {"input": "There is a critical error in the database system.", "result": ""},
        {
            "input": "What are the main findings from the recent market research report and how do they compare to last quarter?",
            "result": "",
        },
    ]
    for inp in test_inputs:
        print("\n--- New Run ---")
        result = workflow.invoke(inp)
        print("Final Result:", result["result"])
