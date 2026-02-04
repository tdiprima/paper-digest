# Main Runner
from agent import agent_run
from database import init_db

if __name__ == "__main__":
    init_db()
    news_sites = [
        "https://feeds.npr.org/1001/rss.xml",  # NPR News RSS feed
        "https://feeds.bbci.co.uk/news/rss.xml",  # BBC News RSS feed
        "https://feeds.reuters.com/Reuters/worldNews",  # Reuters World News RSS
        "https://www.reddit.com/r/worldnews/.rss",  # Reddit World News RSS
    ]
    agent_run(news_sites)
    print("Done scraping and storing news summaries.")
