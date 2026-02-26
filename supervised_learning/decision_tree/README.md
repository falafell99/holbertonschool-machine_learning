# Decision Tree

This directory contains implementations of decision tree algorithms for machine learning.

## Files

### 0-build_decision_tree.py
Contains classes for building a decision tree.

**Classes:**

- **Node**: Represents a node in a decision tree
  - `max_depth_below()`: Calculates maximum depth below the current node
  
- **Leaf**: Represents a leaf node (inherits from Node)
  - `max_depth_below()`: Returns the depth of the leaf
  
- **Decision_Tree**: Represents the entire decision tree
  - `depth()`: Returns the maximum depth of the tree

## Requirements
- Python 3.5+
- pycodestyle 2.5
- numpy

## Usage Examples

```python
from 0-build_decision_tree import Decision_Tree, Node, Leaf

# Create a simple tree
leaf0 = Leaf(0, depth=1)
leaf1 = Leaf(0, depth=2)
leaf2 = Leaf(1, depth=2)
internal_node = Node(feature=1, threshold=30000, 
                     left_child=leaf1, right_child=leaf2, depth=1)
root = Node(feature=0, threshold=0.5, left_child=leaf0, 
            right_child=internal_node, depth=0, is_root=True)
tree = Decision_Tree(root=root)

# Get tree depth
print(tree.depth())  # Output: 2
