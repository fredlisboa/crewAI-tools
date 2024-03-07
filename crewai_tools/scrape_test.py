from tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

from crewai import Agent, Task, Crew, Process

api_gemini = os.environ.get("GEMINI-API-KEY")
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.4, google_api_key=api_gemini
				)

# tool = ScrapeWebsiteTool()

scraper = Agent(
    role='Website Scraper',
    goal=f"""Scrape the given URL.""",
    verbose=True,
    backstory="""Scrape the given URL.""",
    allow_delegation=False,
    max_iter=10,
    max_rpm=20,
    llm=llm_gemini,
    tool=[ScrapeWebsiteTool(website_url='https://https://distribua-alegria.studiodental.care/vcard/persona/PERSONA.html')] 
)

#1-task
scraper_task = Task(
    description=f"""Scrape the given URL.""",
    expected_output="Scrape the given URL.",
    agent=scraper,
    async_execution=False,
)

# Forming the crew with a hierarchical process including the manager
crew = Crew(
    agents=[scraper, ],
    tasks=[scraper_task, ],
    process=Process.sequential,
    manager_llm=llm_gemini,
    verbose=True
)

# Kick off the crew's work
results = crew.kickoff()

# Print the results
print("Crew Work Results:\n")
print(results)