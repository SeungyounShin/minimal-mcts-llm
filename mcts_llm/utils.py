import os
from graphviz import Digraph

def snapshot_mcts(mcts, output_path: str = "mcts_tree_snapshot", max_label_length: int = 30):
    """
    Creates a visual snapshot of the current MCTS tree.

    Args:
        mcts (MCTS): The MCTS object to visualize.
        output_path (str): The file path (without extension) to save the tree visualization.
        max_label_length (int): The maximum length of node labels. Longer labels will be truncated.
    """
    def truncate_label(label: str, max_length: int) -> str:
        """Truncates the label if it exceeds the max length."""
        return label if len(label) <= max_length else label[:max_length - 3] + '...'

    def traverse(node, graph, node_id):
        """
        Recursively traverse the tree and add nodes and edges to the graph.

        Args:
            node (Node): The current node being processed.
            graph (Digraph): The graphviz Digraph object.
            node_id (int): The unique identifier for the current node.

        Returns:
            int: The next available unique node ID.
        """
        current_id = node_id

        # Determine the fill color based on terminal status and value
        if node.is_terminal:
            fillcolor = "lightgreen" if node.value == 1 else "lightcoral"
        else:
            fillcolor = "lightblue"

        truncated_content = truncate_label(node.content, max_label_length)

        graph.node(
            str(current_id),
            label=f"{truncated_content}\nVisits: {node.visit_count}\nValue: {node.value:.2f}",
            shape="box",
            style="filled",
            fillcolor=fillcolor,
        )

        for child in node.children:
            child_id = node_id + 1
            graph.edge(str(current_id), str(child_id))
            node_id = traverse(child, graph, child_id)

        return node_id

    # Initialize the graph
    dot = Digraph(format="png")
    dot.attr(rankdir="TB")  # Top-to-bottom layout

    # Start traversal from the root node
    traverse(mcts.root, dot, 0)

    # Save the visualization to the specified path
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    dot.render(output_path, cleanup=True)
    print(f"MCTS tree snapshot saved to {output_path}.png")
