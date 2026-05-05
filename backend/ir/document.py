"""IR document helpers used by the unified backend pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .types import IRDocument, IRRef, IROperation


@dataclass
class IRDocumentState:
    """Mutable state wrapper for building IR documents incrementally."""

    version: str = "1.0"
    units: str = "mm"
    operations: List[IROperation] = field(default_factory=list)
    entities: Dict[str, List[IRRef]] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    _counter: int = 0

    def next_operation_id(self, prefix: str = "op") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def append(self, operation: IROperation) -> None:
        self.operations.append(operation)

    def extend(self, operations: Iterable[IROperation]) -> None:
        self.operations.extend(list(operations))

    def to_document(self) -> IRDocument:
        return IRDocument(
            version=self.version,
            units="mm",
            operations=list(self.operations),
            entities={key: list(value) for key, value in self.entities.items()},
            metadata=dict(self.metadata) if self.metadata else None,
        )
