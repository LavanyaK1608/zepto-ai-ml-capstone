"""
Zepto Capstone - Module 1: Data Pipeline
Scrape → Clean → Convert → Store → Query
Source: books.toscrape.com
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
from pathlib import Path

BASE_URL = "https://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50
MIN_BOOKS = 60
OUTPUT_DB = Path(__file__).parent / "books.db"
OUTPUT_CSV = Path(__file__).parent / "books_cleaned.csv"

def get_soup(url: str) -> BeautifulSoup:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "html.parser")

def scrape_category(category_url: str, category_name: str) -> list:
    books = []
    page = 1
    while True:
        if page == 1:
            url = category_url
        else:
            url = category_url.replace("index.html", f"page-{page}.html")
        try:
            soup = get_soup(url)
        except Exception:
            break
        articles = soup.select("article.product_pod")
        if not articles:
            break
        for art in articles:
            title = art.h3.a["title"].strip()
            price_text = art.select_one("p.price_color").text.strip()
            rating_class = art.select_one("p.star-rating")["class"]
            star_rating = next((c for c in rating_class if c != "star-rating"), "Zero")
            availability = art.select_one("p.instock.availability").text.strip()
            books.append({
                "title": title,
                "price": price_text,
                "star_rating": star_rating,
                "availability": availability,
                "category": category_name
            })
        next_btn = soup.select_one("li.next a")
        if not next_btn:
            break
        page += 1
        time.sleep(0.3)
    return books

def scrape_all() -> pd.DataFrame:
    soup = get_soup(BASE_URL)
    cat_links = soup.select("div.side_categories ul li ul li a")
    wanted = ["Travel", "Mystery", "Historical Fiction", "Sequential Art",
              "Classics", "Philosophy", "Romance", "Science Fiction"]
    selected = []
    for a in cat_links:
        name = a.text.strip()
        if name in wanted:
            href = a["href"]
            full_url = BASE_URL + href
            selected.append((name, full_url))
    all_books = []
    for name, url in selected[:5]:
        print(f"Scraping category: {name} ...")
        books = scrape_category(url, name)
        print(f"  → {len(books)} books")
        all_books.extend(books)
        if len(all_books) >= MIN_BOOKS:
            break
    df = pd.DataFrame(all_books)
    print(f"\nTotal books scraped: {len(df)}")
    return df

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_gbp"] = df["price"].str.replace(r"[£,]", "", regex=True).astype(float)
    df["rating"] = df["star_rating"].map(RATING_MAP)
    before = len(df)
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with unparseable rating")
    df["in_stock"] = df["availability"].str.contains("In stock", case=False, na=False)
    df = df[["title", "price_gbp", "rating", "in_stock", "category"]].copy()
    return df

def add_inr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    return df

def create_and_load_db(df: pd.DataFrame) -> sqlite3.Connection:
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()
    conn = sqlite3.connect(OUTPUT_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE categories (
            category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE books (
            book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            price_gbp   REAL NOT NULL,
            price_inr   REAL NOT NULL,
            rating      INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            in_stock    INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)
    categories = df["category"].unique()
    for cat in categories:
        cur.execute("INSERT INTO categories (category_name) VALUES (?)", (cat,))
    cat_map = dict(cur.execute("SELECT category_name, category_id FROM categories").fetchall())
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["title"], row["price_gbp"], row["price_inr"],
            int(row["rating"]), 1 if row["in_stock"] else 0,
            cat_map[row["category"]]
        ))
    conn.commit()
    print(f"Database created: {OUTPUT_DB}")
    print(f"  Categories: {len(categories)}")
    print(f"  Books: {len(df)}")
    return conn

QUERIES = {
    "1_highest_price": """
        SELECT title, price_gbp, price_inr, rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 5;
    """,
    "2_in_stock_high_rating": """
        SELECT title, rating, price_inr
        FROM books
        WHERE in_stock = 1 AND rating >= 4
        ORDER BY rating DESC, price_inr ASC
        LIMIT 10;
    """,
    "3_distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "4_price_between": """
        SELECT title, price_gbp, category_id
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,
    "5_join_top_rated_per_category": """
        SELECT c.category_name,
               b.title,
               b.rating,
               b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating = 5
        ORDER BY c.category_name, b.price_inr DESC
        LIMIT 15;
    """
}

def run_sql_queries(conn: sqlite3.Connection):
    print("\n" + "="*60)
    print("SQL QUERY RESULTS")
    print("="*60)
    results = {}
    for name, sql in QUERIES.items():
        print(f"\n--- {name} ---")
        print(sql.strip())
        dfq = pd.read_sql(sql, conn)
        print(dfq.to_string(index=False))
        results[name] = dfq
    return results

def pandas_equivalents(df: pd.DataFrame, sql_results: dict):
    print("\n" + "="*60)
    print("PANDAS EQUIVALENTS (for the JOIN query)")
    print("="*60)
    categories = df[["category"]].drop_duplicates().reset_index(drop=True)
    categories["category_id"] = categories.index + 1
    categories = categories.rename(columns={"category": "category_name"})
    books = df.copy()
    books = books.merge(categories, left_on="category", right_on="category_name")
    books = books.drop(columns=["category", "category_name"])
    pandas_join = (
        books[books["rating"] == 5]
        .merge(categories, on="category_id")
        .sort_values(["category_name", "price_inr"], ascending=[True, False])
        [["category_name", "title", "rating", "price_inr"]]
        .head(15)
        .reset_index(drop=True)
    )
    sql_join = sql_results["5_join_top_rated_per_category"].reset_index(drop=True)
    print("\nSQL result (JOIN):")
    print(sql_join.to_string(index=False))
    print("\npandas merge result:")
    print(pandas_join.to_string(index=False))
    match = sql_join.equals(pandas_join)
    print(f"\nResults match: {match}")
    return pandas_join

def main():
    print("=== Zepto Capstone – Data Pipeline ===\n")
    raw_df = scrape_all()
    if len(raw_df) < MIN_BOOKS:
        raise RuntimeError(f"Only scraped {len(raw_df)} books – need ≥ {MIN_BOOKS}")
    clean_df = clean_data(raw_df)
    print(f"After cleaning: {len(clean_df)} rows")
    final_df = add_inr(clean_df)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Cleaned CSV saved: {OUTPUT_CSV}")
    conn = create_and_load_db(final_df)
    sql_results = run_sql_queries(conn)
    pandas_equivalents(final_df, sql_results)
    conn.close()
    print("\n✅ Data pipeline finished successfully.")

if __name__ == "__main__":
    main()
