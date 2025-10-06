"""
The Agent[None, str]
  annotation ensures type safety where None represents no dependencies 
  and str represents the expected output return type.
"""

from pydantic_ai import Agent

# Create an agent with a system prompt and response model validation
agent: Agent[None, str] = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="Give me a 1-sentence description of PydanticAI.",
)

# Run the agent
result = agent.run_sync("")
print(result.output)
