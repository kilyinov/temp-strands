"""The profile graph: plan -> {housing, healthcare, economy, education} in parallel -> verify."""

from __future__ import annotations

from strands.multiagent import GraphBuilder
from strands.multiagent.graph import Graph, GraphState

from ..models import TOPICS
from ..telemetry import NodeTimingHook
from .nodes import PlanNode, TopicAnalystNode, VerifyNode

ANALYST_IDS = tuple(t.value for t in TOPICS)


def _all_analysts_done(state: GraphState) -> bool:
    done = {n.node_id for n in state.completed_nodes}
    return all(a in done for a in ANALYST_IDS)


def build_graph(execution_timeout: float = 900.0) -> Graph:
    builder = GraphBuilder()
    builder.add_node(PlanNode(), "plan")
    for topic in TOPICS:
        builder.add_node(TopicAnalystNode(topic), topic.value)
        builder.add_edge("plan", topic.value)
    builder.add_node(VerifyNode(), "verify")
    for analyst in ANALYST_IDS:
        builder.add_edge(analyst, "verify", condition=_all_analysts_done)
    builder.set_entry_point("plan")
    builder.set_execution_timeout(execution_timeout)
    builder.set_max_node_executions(len(ANALYST_IDS) + 2)
    builder.set_hook_providers([NodeTimingHook()])
    return builder.build()
