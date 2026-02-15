"""
Among Us AI agent framework.

Public API
----------
::

    from agents import BaseAgent, OpenAIAgent, Role, ActionType
    from agents import Message, EmergencyMeeting, SystemEvent, Action
    from agents import Interaction, AgentContext
    from agents import EnvAgentAdapter
"""

from agents.base_agent import BaseAgent
from agents.models import (
    Action,
    ActionType,
    AgentContext,
    EmergencyMeeting,
    Interaction,
    Message,
    Role,
    SystemEvent,
)
from agents.openai_agent import OpenAIAgent
from agents.openrouter_agent import OpenRouterAgent
from agents.env_adapter import EnvAgentAdapter

__all__ = [
    "BaseAgent",
    "OpenAIAgent",
    "OpenRouterAgent",
    "EnvAgentAdapter",
    "Role",
    "ActionType",
    "Message",
    "EmergencyMeeting",
    "SystemEvent",
    "Action",
    "Interaction",
    "AgentContext",
]
