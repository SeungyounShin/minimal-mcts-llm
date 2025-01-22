import math

class Node:
    def __init__(self, content : str, parent = None, is_terminal = False):
        self.content = content
        self.children = []
        self.parent = parent
        self.visit_count = 0    
        self.value = 0
        self.total_reward = 0
        self.is_terminal = is_terminal
    
    def __repr__(self):
        return f"Node(\n\tcontent={self.content},\n\tvisit={self.visit_count},\n\tvalue={self.value}\n)"

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def update_value(self):
        self.value = self.total_reward / self.visit_count
    
    def get_children_and_not_terminal(self):
        return [child for child in self.children if not child.is_terminal]

    def _ucb1(self, c: float = 1.41):
        """
        Calculate the UCB1 value for a node.
        """
        if self.visit_count == 0:
            return float('inf')  # Favor unexplored nodes
        
        if self.parent is None or self.parent.visit_count == 0:
            return 0  # This shouldn't happen for a valid MCTS tree, but safeguard

        exploitation = self.value
        # Prevent log(0) and division by zero
        exploration = math.sqrt(
            max(0, math.log(self.parent.visit_count) / self.visit_count)
        )
        
        return exploitation + c * exploration
