from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass
class SessionMemory:
    confirmed_traits: dict[str, Any] = field(default_factory=dict)
    soft_dispreferred: list[str] = field(default_factory=list)
    rejected_hypotheses: list[dict[str, str]] = field(default_factory=list)
    player_details: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMemory":
        return cls(**data)
        
    def summary_prompt(self) -> str:
        """Format clean prompt for Gemma-26B deduction."""
        lines = []
        if self.confirmed_traits:
            lines.append("Confirmed facts: " + ", ".join(f"{k}: {v}" for k, v in self.confirmed_traits.items()))
        if self.player_details:
            lines.append("Player details: " + "; ".join(self.player_details))
        if self.soft_dispreferred:
            lines.append("Player answered NO to: " + ", ".join(self.soft_dispreferred[:8]))
        if self.rejected_hypotheses:
            lines.append("Already rejected: " + ", ".join(f"{r.get('name', '')} ({r.get('series', '')})" for r in self.rejected_hypotheses))
        return "\n".join(lines)
