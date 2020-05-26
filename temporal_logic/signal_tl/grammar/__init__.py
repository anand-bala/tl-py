"""
This subpackage defines the grammar for STL.

Only the most basic abstractions are defined. To evaluate a formula or monitor a signal with a spec, 2 methods can be
used:

1. Use in-built algorithms from the signal_tl.functional package; or
2. Define custom algorithms.

The idea is to leverage Python's dynamic typing to create a "syntax-tree"-like structure on which one can perform
recursive iteration and implement algorithms.

"""
