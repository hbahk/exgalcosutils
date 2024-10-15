"""query_arxiv.py: Query arXiv submissions."""

import datetime
import sys
import urllib

import feedparser
import pytz
from pandas.tseries.offsets import BDay


def get_query_url(
    catcondi: str, datecondi: str, start: int = -1, max_results: int = -1
) -> str:
    """Get query url for arXiv submissions.

    Args:
        catcondi: Category condition.
        datecondi: Date condition.
        start: Starting index of the results. If -1, it will start from the beginning.
            Default is -1.
        max_results: Maximum number of results to retrieve. If -1, it will retrieve all
            results. Default is -1.

    Returns:
        str: Query string.
    """
    # Base api query url
    base_url = "http://export.arxiv.org/api/query?"

    # Search parameters
    search_query = f"{catcondi}+AND+{datecondi}"

    query = f"search_query={search_query}&sortBy=submittedDate"

    if start != -1:
        query += f"&start={start}"
    if max_results != -1:
        query += f"&max_results={max_results}"

    return base_url + query


def get_catcondi(categories: list[str]) -> str:
    """Get category condition for arXiv submissions.

    Args:
        categories: List of categories.

    Returns:
        str: Category condition.
    """
    if len(categories) == 0:
        catcondi = ""
    else:
        catcondi = build_nested_conditions([f"cat:{cat}" for cat in categories])

    return catcondi


def build_nested_conditions(conditions: list[str]) -> str:
    """Builds a nested OR condition where each pair of conditions is enclosed in parentheses.

    Args:
        conditions: List of condition strings already prefixed with 'cat:'.

    Returns:
        str: Nested OR condition.
    """
    while len(conditions) > 1:
        # Pair up and enclose conditions
        temp_conditions = []
        for i in range(0, len(conditions), 2):
            if i + 1 < len(conditions):
                combined = f"%28{conditions[i]}+OR+{conditions[i+1]}%29"
            else:
                combined = conditions[i]  # Odd condition out
            temp_conditions.append(combined)
        conditions = temp_conditions
    return conditions[0]


def get_datecondi(start_date: datetime.datetime, end_date: datetime.datetime) -> str:
    """Get date condition for arXiv submissions.

    Args:
        start_date: Start date.
        end_date: End date.

    Returns:
        str: Date condition.
    """
    start_utc = start_date.astimezone(pytz.utc)
    end_utc = end_date.astimezone(pytz.utc)

    datecondi = f"submittedDate:[{start_utc.strftime('%Y%m%d%H%M')}+TO+{end_utc.strftime('%Y%m%d%H%M')}]"

    return datecondi


def get_datecondi_daily(date: datetime.datetime) -> str:
    """Get date condition for arXiv submissions for a single day.

    For a given date, it will retrieve the submissions from the previous two business
    days. Since the arXiv submissions are updated at 18:00 UTC, the start and end dates
    are set to 18:00 UTC.

    Args:
        date: Date. If the timezone is not set, it will be set to the local timezone.

    Returns:
        str: Date condition.
    """
    if date.tzinfo is None:
        date = date.astimezone()

    date = date.replace(hour=12, minute=0, second=0, microsecond=0)  # initializing
    date_utc = date.astimezone(pytz.utc)

    end = date_utc - BDay(1)  # Business day before the given date
    start = end - BDay(1)  # Two Business days before the given date

    start_date = start.replace(hour=18, minute=0, second=0, microsecond=0)
    end_date = end.replace(hour=18, minute=0, second=0, microsecond=0)

    return get_datecondi(start_date, end_date)


def query_url(url: str) -> feedparser.FeedParserDict:
    """Query arXiv submissions using the given url.

    Args:
        url: Query url.

    Returns:
        feedparser.FeedParserDict: Feed information.
    """

    # Opensearch metadata such as totalResults, startIndex, and itemsPerPage live
    # in the opensearch namespase. Some entry metadata lives in the arXiv namespace.
    # This is a hack to expose both of these namespaces in feedparser v6.0.11
    feedparser.mixin._FeedParserMixin.namespaces[
        "http://a9.com/-/spec/opensearch/1.1/"
    ] = "opensearch"
    feedparser.mixin._FeedParserMixin.namespaces["http://arxiv.org/schemas/atom"] = (
        "arxiv"
    )

    # perform a GET request using the base_url and query
    response = urllib.request.urlopen(url).read()

    # parse the response using feedparser
    feed = feedparser.parse(response)

    return feed


def query_arxiv(
    categories: list[str],
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    start: int = -1,
    max_results: int = -1,
) -> feedparser.FeedParserDict:
    """Query arXiv submissions.

    Args:
        categories: List of categories.
        start_date: Start date.
        end_date: End date.
        start: Starting index of the results. If -1, it will start from the beginning.
            Default is -1.
        max_results: Maximum number of results to retrieve. If -1, it will retrieve all
            results. Default is -1.

    Returns:
        feedparser.FeedParserDict: Feed information.
    """
    catcondi = get_catcondi(categories)
    datecondi = get_datecondi(start_date, end_date)
    url = get_query_url(catcondi, datecondi, start, max_results)

    return query_url(url)


def query_arxiv_daily(
    categories: list[str],
    date: datetime.datetime,
    start: int = -1,
    max_results: int = -1,
) -> feedparser.FeedParserDict:
    """Query arXiv submissions for a single day.

    Args:
        categories: List of categories.
        date: Date. If the timezone is not set, it will be set to the local timezone.
        start: Starting index of the results. If -1, it will start from the beginning.
            Default is -1.
        max_results: Maximum number of results to retrieve. If -1, it will retrieve all
            results. Default is -1.

    Returns:
        feedparser.FeedParserDict: Feed information.
    """
    datecondi = get_datecondi_daily(date)
    catcondi = get_catcondi(categories)
    url = get_query_url(catcondi, datecondi, start, max_results)

    return query_url(url)


def print_titles(feed: feedparser.FeedParserDict):
    """Print titles of the feed entries.

    Args:
        feed: Feed information.
    """
    for entry in feed.entries:
        print('* ' + entry.title)


def save_feed(feed: feedparser.FeedParserDict, filename: str, include_abstract=False):
    """Save feed information to a file.

    Args:
        feed: Feed information.
        filename: File name.
    """
    with open(filename, 'w') as f:
        for entry in feed.entries:
            mdentry = make_markdown_entry(entry, include_abstract)
            f.write(mdentry)


def make_markdown_entry(
    entry: feedparser.FeedParserDict, include_abstract=False
) -> str:
    """Make a markdown entry for the feed entry.

    Args:
        entry: a single feed entry.
        include_abstract: If True, include the abstract in the markdown entry.
            Default is False.

    Returns:
        str: Markdown entry.
    """
    # entry_id = entry.id.split("/abs/")[-1]
    title = entry.title.replace("\n", "")
    authors = [author.name for author in entry.authors]
    # published = entry.published
    abs_link = next(link.href for link in entry.links if link.type == "text/html")
    pdf_link = next(link.href for link in entry.links if link.type == "application/pdf")
    # tags = [tag.term for tag in entry.tags]

    last_names = [author.split()[-1] for author in authors]
    if len(authors) == 1:
        authstr = last_names[0]
    elif len(authors) == 2:
        authstr = r" \& ".join(last_names)
    else:
        authstr = last_names[0] + " et al."

    markdown = f"* [[abs]({abs_link})][[pdf]({pdf_link})] **{title}** ({authstr})\n"
    if include_abstract:
        abstract = entry.summary.replace("\n", " ")
        markdown += f"  - {abstract}\n"

    return markdown


if __name__ == "__main__":
    categories = ["astro-ph.GA", "astro-ph.CO"]
    include_abstract = False

    # Check if a command line argument is provided for the date
    if len(sys.argv) > 1:
        date_input = sys.argv[1]
        try:
            # Try to convert the input string to a date
            specific_date = datetime.datetime.strptime(date_input, "%Y-%m-%d")
        except ValueError:
            # If there is an error in date conversion, use today's date
            print("Invalid date format. Using today's date.")
            specific_date = datetime.datetime.now()
    else:
        # Default to today's date if no argument is provided
        specific_date = datetime.datetime.now()

    if len(sys.argv) > 2:
        include_abstract = "--include-abstract" in sys.argv

    feed = query_arxiv_daily(categories, specific_date, max_results=100)
    print_titles(feed)
    print(f"Total results: {feed.feed.opensearch_totalresults}")
    if include_abstract:
        save_feed(
            feed, f"arxiv_abs_{specific_date.strftime('%Y-%m-%d')}.md", include_abstract
        )
    else:
        save_feed(feed, f"arxiv_{specific_date.strftime('%Y-%m-%d')}.md")
