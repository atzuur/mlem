from dataclasses import dataclass, field
import pydot

from . import utils
from .dataset import Dataset


@dataclass
class DecisionNode:
    status: str = None
    descendants: dict[str, "DecisionNode"] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return not self.descendants


class ID3Classifier:
    root: DecisionNode
    branch: DecisionNode
    
    def __init__(self):
        self.branch = self.root = DecisionNode()

    def fit(self, dataset: Dataset):
        if dataset.target.entropy() == 0:
            self.branch.status = dataset.target[0]
            return
        if not dataset:
            self.branch.status = dataset.target.mode()
            return

        best_attr = dataset.most_informative_attr()
        self.branch.status = best_attr

        old_branch = self.branch
        for split_value, data_part in dataset.split(best_attr).items():
            self.branch.descendants[split_value] = DecisionNode()
            self.branch = self.branch.descendants[split_value]
            self.fit(data_part)
            self.branch = old_branch

        return self

    def predict(self, data: dict[str, list[str]]) -> str:
        def predict_sample(
            sample: dict[str, str],
            branch: DecisionNode = self.root
        ) -> bool:
            if branch.is_leaf():
                return branch.status
            next_branch = branch.descendants[sample[branch.status]]
            return predict_sample(sample, next_branch)

        # all columns should have equal length
        n_samples = len(data[list(data.keys())[0]])
        return [predict_sample(utils.row_of_doa(data, i)) for i in range(n_samples)]

    def tree_to_dot(self, graph_name: str) -> str:
        graph = pydot.Graph(graph_name)

        def add_node(branch: DecisionNode, node_name: str = ""):
            graph.add_node(pydot.Node(
                node_name := node_name or branch.status,
                label=branch.status,
                shape="plaintext" if branch.is_leaf() else "ellipse"
            ))
            for edge, child in branch.descendants.items():
                child_name = f"{node_name}_{edge}"
                graph.add_edge(pydot.Edge(node_name, child_name, label=edge))
                add_node(child, child_name)

        add_node(self.root)

        return graph.to_string()
