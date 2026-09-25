from dataclasses import dataclass, field

"""
TXT  → ContentElement(type="text")
MD   → ContentElement(type="text"/"heading"/...)
PDF  → ContentElement(type="text"/"table"/"image")
"""


@dataclass
class ContentElement:
    type: str
    content: str
    metadata: dict[str, object] = field(default_factory=dict)
