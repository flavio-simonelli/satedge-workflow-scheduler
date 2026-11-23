"""Routing of a data item around the reservations: the types, the interface and the router.

A router finds, for a data item, the routes to the nodes it can reach, around the reservations
already made (:class:`~satedge_workflow_scheduler.solvers.sequential.reservations.Reservations`).
The router is the only part of the family that decides how data travel; the placement only
uses its routes.
"""
