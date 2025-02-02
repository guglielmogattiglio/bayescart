"""Custom exceptions for bayescart.

This module defines exceptions used by bayescart to signal errors
in tree operations and abstract method implementations.
"""

class NotImplementedError(Exception):
    """Exception raised when an abstract method has not been implemented."""
    pass

class InvalidTreeError(Exception):
    """Exception raised for errors in the tree structure or data."""
    pass

class AbstractMethodError(Exception):
    """Exception raised when an abstract method has not been implemented."""
    pass
