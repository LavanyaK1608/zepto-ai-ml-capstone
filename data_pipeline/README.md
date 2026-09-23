# Module 1 – Data Pipeline (25 marks)

## Purpose
Scrapes live book catalogue data from books.toscrape.com, cleans it, converts GBP → INR with the project-fixed rate, stores it in a normalized SQLite database, and demonstrates SQL + pandas querying.

## Setup
```bash
pip install requests beautifulsoup4 pandas
```

## How to run
```bash
cd data_pipeline
python scrape_and_load.py
```

This will:
1. Scrape ≥ 60 books across ≥ 3 categories
2. Clean fields → `price_gbp`, `rating` (1-5), `in_stock` (bool)
3. Add `price_inr` using **1 GBP = 105.50 INR** (fixed project constant)
4. Create `books.db` with two tables (`categories` ↔ `books`) linked by FK
5. Execute 5 SQL queries covering SELECT/WHERE/ORDER BY/LIMIT/DISTINCT/BETWEEN + JOIN
6. Reproduce the JOIN result with `pd.merge` and show they match

## Design decisions
- **Parsing failures**: Any row with an unrecognised star-rating text is dropped (very rare on this site). Median imputation was not needed.
- **Currency**: Only the required fixed rate `105.50` is used. No live API.
- **Schema**: Classic 2-table design – `categories` (dimension) and `books` (fact) with a proper foreign key.
- **Availability**: Converted to boolean `in_stock` by simple string contains check.

## Files
- `scrape_and_load.py` – complete pipeline
- `books.db` – generated SQLite database
- `books_cleaned.csv` – cleaned intermediate CSV (optional convenience)
