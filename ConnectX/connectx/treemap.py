from __future__ import annotations
from typing import Dict, Callable, Optional


class TreeMap:

    def __init__(self, value=None, parent: Optional[TreeMap] = None):
        self._branches: Dict[object, TreeMap] = {}
        self._value = value
        self._parent = parent

    @property
    def value(self):
        """ The value of this node. """
        return self._value

    @property
    def key(self):
        """ They key on the parent that has this branch. """
        if self.parent is None:
            return None

        for key, branch in self.parent.branches.items():
            if branch == self:
                return key

    @property
    def parent(self):
        """ The parent of this branch. """
        return self._parent

    @property
    def branches(self):
        return self._branches

    @property
    def depth(self):
        """ Returns the depth of this branch. """
        counter = 0
        parent = self._parent
        while parent is not None:
            counter += 1
            parent = parent.parent
        return counter

    def add(self, key, value) -> TreeMap:
        """ Adds a new branch and returns the new branch. """
        if key in self.branches:
            raise Exception(f"Cannot add branch because key {key} already exists.")

        self._branches[key] = TreeMap(value, self)
        return self._branches[key]

    def add_or_update_branch(self, key, value, update_value) -> TreeMap:
        """
        Creates a new branch with the given key and value or updates the value at the branch
        with the given key.
        :returns the branch with the key
        """
        if key in self.branches:
            branch = self.branches[key]
            branch._value = update_value(branch._value)
            return branch
        else:
            return self.add(key, value)

    def update_and_propagate_up_tree(self, update_node: Callable[[object], object]):
        """ Updates the value with the given function and apply the same function to all nodes up the tree. """
        self._value = update_node(self._value)
        if self._parent is not None:
            self._parent.update_and_propagate_up_tree(update_node)

    def __repr__(self):
        return f'Depth: {self.depth}, value: {self._value}, keys: {self.branches.keys()}'


def test_tree_map():
    import numpy as np

    branch = TreeMap(0)
    for _ in range(10):
        branch = branch.add_or_update_branch(np.random.randint(0, 7), np.random.randint(-1, 2), None)

    while branch is not None:
        print(branch)
        branch = branch.parent


