"""
This represents what PydanticAI actually does - creating type-safe AI agents with structured outputs.

1. Agent creation with structured output using Agent class
2. Type-safe responses with a PersonalizedGreeting Pydantic model
3. Async/await pattern which is core to PydanticAI
4. Structured data extraction from LLM responses
5. Type validation and safety guarantees
"""

from pydantic import BaseModel
from pydantic_ai import Agent


# Define a structured response model
class PersonalizedGreeting(BaseModel):
    greeting: str
    language: str
    enthusiasm_level: int  # 1-10 scale


# Create an agent with structured output
greeting_agent = Agent(
    "openai:gpt-4o-mini",
    result_type=PersonalizedGreeting,
    system_prompt="You are a friendly greeting assistant. Generate personalized greetings with varying enthusiasm levels.",
)


async def main():
    # Run the agent with a user prompt
    result = await greeting_agent.run(
        "Generate a greeting for a Python developer who just discovered PydanticAI"
    )

    # The result is automatically validated and typed
    greeting_data = result.data
    print(f"Greeting: {greeting_data.greeting}")
    print(f"Language: {greeting_data.language}")
    print(f"Enthusiasm: {greeting_data.enthusiasm_level}/10")

    # Demonstrate type safety - this is a PersonalizedGreeting object
    assert isinstance(greeting_data, PersonalizedGreeting)
    print(f"\nFull response: {greeting_data.model_dump()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
