"""Application of a contact plan to the links of an instance document."""

from __future__ import annotations

from satedge_workflow_scheduler.io.documents import ContactPlanDoc, InstanceDoc
from satedge_workflow_scheduler.io.errors import DocumentError
from satedge_workflow_scheduler.io.translate.instance import explicit_topology


def apply_contact_plan(doc: InstanceDoc, plan: ContactPlanDoc) -> InstanceDoc:
    """The explicit instance document with the windows of ``plan`` on its links.

    A link that the plan names gets those windows. The plan may only name links of the
    topology, each once, and only links whose windows are not written in the instance already.

    Raises:
        DocumentError: If the plan names a missing link, names a link twice, or a link that
            already has windows in the instance.
    """
    topology = explicit_topology(doc)
    links = list(topology.links)
    position = {frozenset((link.a, link.b)): i for i, link in enumerate(links)}
    named: set[frozenset[str]] = set()
    for entry in plan.links:
        key = frozenset((entry.a, entry.b))
        label = f"{entry.a}-{entry.b}"
        if key not in position:
            raise DocumentError(f"contact plan: the topology has no link {label}")
        if key in named:
            raise DocumentError(f"contact plan: link {label} is listed twice")
        index = position[key]
        if links[index].windows is not None:
            raise DocumentError(
                f"link {label} is windowed in the instance: remove its windows or its "
                "contact plan entry"
            )
        links[index] = links[index].model_copy(
            update={"windows": entry.windows, "windowed": True}
        )
        named.add(key)
    new_topology = topology.model_copy(update={"links": links})
    return doc.model_copy(update={"topology": new_topology})
