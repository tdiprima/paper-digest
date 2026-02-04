# python3.10 run_stock_alert.py
import smtplib
from email.mime.text import MIMEText

from config import (EMAIL_PASSWORD, EMAIL_RECEIVER, EMAIL_SENDER,
                    PRICE_CHANGE_THRESHOLD, STOCK_SYMBOLS)
from crewai import Agent, Crew, Task


# Tool: Email sender
def send_email_alert(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())


# Create CrewAI Agents
researcher = Agent(
    role="Stock Price Researcher",
    goal="Fetch and monitor current stock prices for specified symbols",
    backstory="""You are a diligent financial data researcher who specializes in 
    retrieving accurate, up-to-date stock price information. You ensure data quality 
    and provide reliable market data for analysis.""",
    # verbose=True,
    allow_delegation=False,
)

analyst = Agent(
    role="Stock Price Analyst",
    goal="Analyze stock price changes and identify significant movements",
    backstory="""You are an experienced financial analyst who specializes in 
    identifying significant stock price movements. You calculate percentage changes 
    and determine when alerts should be triggered based on predefined thresholds.""",
    # verbose=True,
    allow_delegation=False,
)


# Create CrewAI Tasks
def create_research_task():
    return Task(
        description=f"""Fetch current and previous stock prices for the following symbols: {', '.join(STOCK_SYMBOLS)}.
        For each symbol, retrieve:
        1. Previous closing price
        2. Current price
        
        Return the data in a structured format that can be used for analysis.""",
        agent=researcher,
        expected_output="Stock price data with previous and current prices for each symbol",
    )


def create_analysis_task():
    return Task(
        description=f"""Analyze the stock price data and:
        1. Calculate percentage change for each stock
        2. Identify stocks with changes >= {PRICE_CHANGE_THRESHOLD}%
        3. Generate insights and alerts for significant changes
        4. If significant changes are detected, prepare email alerts
        
        Use the stock price data from the research task.""",
        agent=analyst,
        expected_output="Analysis report with insights, alerts, and email notifications if needed",
    )


def main():
    print("🚀 Starting Stock Alert System with CrewAI...")
    print("=" * 50)

    # Create tasks
    research_task = create_research_task()
    analysis_task = create_analysis_task()

    # Create and execute crew
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        # verbose=True
    )

    # Execute the crew
    result = crew.kickoff()

    print("\n" + "=" * 50)
    print("🏁 CrewAI Stock Alert System completed.")
    print(f"Final Result: {result}")


if __name__ == "__main__":
    main()
