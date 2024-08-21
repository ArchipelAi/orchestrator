from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import START, Graph

from orchestrator.steps.execute_step.execute_step import execute_step
from orchestrator.steps.plan_step.plan_step import plan_step

workflow = Graph()

workflow.add_node('planner', plan_step)
workflow.add_node('executor', execute_step)

workflow.add_edge(START, 'planner')
workflow.add_edge('planner', 'executor')
workflow.add_edge('executor', 'executor')

app = workflow.compile()


def main():
    with open('graph.png', 'wb') as fp:
        fp.write(
            app.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )


if __name__ == '__main__':
    main()
