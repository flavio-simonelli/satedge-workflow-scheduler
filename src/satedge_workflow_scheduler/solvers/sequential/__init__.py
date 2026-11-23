"""Sequential constructive heuristics: order the requests, then plan one request at a time.

The family has a fixed skeleton (see :mod:`.solver`): sort the new requests, then plan each one
around the reservations of the requests planned before it. What changes between the members of
the family is the order (:mod:`.ordering`), how a request is placed (:mod:`.placement`), and how
its data are routed (:mod:`.routing`).

This package holds the parts that do not depend on those choices: the reservations.
"""
