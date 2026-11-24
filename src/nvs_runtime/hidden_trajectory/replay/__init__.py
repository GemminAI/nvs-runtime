"""Recomputation of kernel stages from a stored K record (Phase 4, method B).

A subpackage of ``hidden_trajectory`` on purpose (it needs numpy, which the modules of
``hidden_trajectory/`` itself do not import: ``test_e09``). Nothing of the capture or the R route
imports from here.
"""
