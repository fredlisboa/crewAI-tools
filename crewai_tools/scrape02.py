import os

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

from tools.selenium_scraping_tool.selenium_scraping_tool import SeleniumScrapingTool

from crewai import Agent, Task, Crew

api_gemini = os.environ.get("GEMINI-API-KEY")
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.4, google_api_key=api_gemini
				)

print("## Welcome to the scraper Crew")
print('-------------------------------\n')
website_url = input("What is the URL website you want to scrape?\n")


# Instantiate tools
scrape_tool = SeleniumScrapingTool()

# Create agents
scraper = Agent(
    role='Scrape an URL.',
    goal='Scrape and summarize an URL.',
    backstory='An expert analyst with a keen eye scrape and summarize.',
    verbose=True,
    llm=llm_gemini,
    tools=[scrape_tool]
)

# Define tasks
scraper_task = Task(
    description=f"""Scrape the following URL: {website_url}""",
    expected_output='Summarize URL website content.',
    agent=scraper,
    )

# Assemble a crew
crew = Crew(
    agents=[scraper],
    tasks=[scraper_task],
    verbose=2
)

# Execute tasks
crew.kickoff()