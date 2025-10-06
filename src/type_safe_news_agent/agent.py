# News Scraping Agent
from datetime import datetime
from typing import List

import requests
from bs4 import BeautifulSoup

from database import save_news_summary
from models import NewsSummary


def fetch_articles_from_url(url: str) -> List[NewsSummary]:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "xml")
    articles = []

    # Handle RSS feeds
    for item in soup.find_all("item"):
        try:
            title_tag = item.find("title")
            link_tag = item.find("link")
            description_tag = item.find("description")
            pubdate_tag = item.find("pubDate")

            # Extract summary from description (remove HTML tags if present)
            summary_text = (
                description_tag.get_text(strip=True)
                if description_tag
                else "No Summary"
            )
            if description_tag:
                desc_soup = BeautifulSoup(summary_text, "html.parser")
                summary_text = desc_soup.get_text(strip=True)

            # Parse publication date
            pub_date = datetime.now()
            if pubdate_tag:
                try:
                    from email.utils import parsedate_to_datetime

                    pub_date = parsedate_to_datetime(pubdate_tag.get_text(strip=True))
                except Exception:
                    pub_date = datetime.now()

            news = NewsSummary(
                title=title_tag.get_text(strip=True) if title_tag else "No Title",
                url=link_tag.get_text(strip=True) if link_tag else url,
                summary=(
                    summary_text[:500] + "..."
                    if len(summary_text) > 500
                    else summary_text
                ),
                published_at=pub_date,
                source=url,
            )
            articles.append(news)
        except Exception as e:
            print(f"Skipping article due to error: {e}")

    # Fallback for non-RSS content (original HTML scraping)
    if not articles:
        soup = BeautifulSoup(response.text, "html.parser")
        for item in soup.select("article"):
            try:
                title_tag = item.find("h2")
                link_tag = item.find("a", href=True)
                summary_tag = item.find("p")
                date_tag = item.find("time")

                news = NewsSummary(
                    title=title_tag.get_text(strip=True) if title_tag else "No Title",
                    url=link_tag["href"] if link_tag else url,
                    summary=(
                        summary_tag.get_text(strip=True)
                        if summary_tag
                        else "No Summary"
                    ),
                    published_at=(
                        datetime.fromisoformat(date_tag["datetime"])
                        if date_tag and date_tag.has_attr("datetime")
                        else datetime.now()
                    ),
                    source=url,
                )
                articles.append(news)
            except Exception as e:
                print(f"Skipping article due to error: {e}")

    return articles


def agent_run(news_sites: List[str]):
    for site in news_sites:
        print(f"Scraping {site}")
        summaries = fetch_articles_from_url(site)
        for summary in summaries:
            save_news_summary(summary)
