import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent("assistant", model_client=model_client)
    print(await agent.run(task="Give a 1-sentence explanation of when to use AutoGen."))
    await model_client.close()


asyncio.run(main())
