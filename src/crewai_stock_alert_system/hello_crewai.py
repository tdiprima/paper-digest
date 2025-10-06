"""
A simple example of using CrewAI to create a friendly greeting for a user.
"""

from crewai import Agent, Crew, Task

# Define an AI Agent
# This agent is a "Greeter" whose role is to create friendly greetings.
greeter_agent = Agent(
    role="Greeter",
    goal='Generate a friendly "Hello, World!" message personalized for the user',
    backstory="You are a cheerful AI assistant who loves starting conversations with warm greetings.",
    # verbose=True,  # Enable detailed logging
    allow_delegation=False,  # No need for delegation in this simple example
    # You can add tools here if needed, but for simplicity, we'll skip them
)

# Define a Task for the agent
# The task is to create a greeting message.
greeting_task = Task(
    description='Create a personalized "Hello, World!" greeting for a user named Somebody. Make it fun and engaging.',
    expected_output="A string containing the greeting message.",
    agent=greeter_agent,
)

# Create a Crew (a group of agents working on tasks)
# In this case, it's a crew with just one agent and one task.
hello_crew = Crew(
    agents=[greeter_agent],
    tasks=[greeting_task],
    # verbose=2  # 1 for minimal logging, 2 for detailed
)

# Kick off the crew and execute the task
result = hello_crew.kickoff()

# Print the result
print("CrewAI Hello World Result:")
print(result)
