"""Create and populate databases/bookstore.sqlite.

Run from repo root:
    python databases/create_bookstore.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "bookstore.sqlite"

SCHEMA = """
CREATE TABLE authors (
    author_id   INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    nationality TEXT,
    birth_year  INTEGER
);

CREATE TABLE books (
    book_id       INTEGER PRIMARY KEY,
    title         TEXT    NOT NULL,
    genre         TEXT,
    year_published INTEGER,
    price         REAL,
    author_id     INTEGER REFERENCES authors(author_id)
);

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT,
    city         TEXT,
    member_since INTEGER
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER REFERENCES customers(customer_id),
    order_date   TEXT,
    total_amount REAL
);

CREATE TABLE order_items (
    item_id    INTEGER PRIMARY KEY,
    order_id   INTEGER REFERENCES orders(order_id),
    book_id    INTEGER REFERENCES books(book_id),
    quantity   INTEGER,
    unit_price REAL
);
"""

AUTHORS = [
    (1, "George Orwell",        "British",    1903),
    (2, "J.K. Rowling",         "British",    1965),
    (3, "Frank Herbert",         "American",   1920),
    (4, "Ursula K. Le Guin",    "American",   1929),
    (5, "Cormac McCarthy",       "American",   1933),
    (6, "Haruki Murakami",       "Japanese",   1949),
    (7, "Gabriel García Márquez","Colombian",  1927),
    (8, "Kazuo Ishiguro",        "British",    1954),
    (9, "Toni Morrison",         "American",   1931),
    (10,"Philip K. Dick",        "American",   1928),
]

BOOKS = [
    (1,  "1984",                              "Dystopian",      1949, 12.99,  1),
    (2,  "Animal Farm",                       "Political",      1945,  9.99,  1),
    (3,  "Harry Potter and the Philosopher's Stone", "Fantasy", 1997, 14.99,  2),
    (4,  "Harry Potter and the Chamber of Secrets",  "Fantasy", 1998, 14.99,  2),
    (5,  "Dune",                              "Science Fiction",1965, 16.99,  3),
    (6,  "Dune Messiah",                      "Science Fiction",1969, 14.99,  3),
    (7,  "The Left Hand of Darkness",         "Science Fiction",1969, 13.99,  4),
    (8,  "The Dispossessed",                  "Science Fiction",1974, 12.99,  4),
    (9,  "The Road",                          "Post-Apocalyptic",2006,11.99,  5),
    (10, "No Country for Old Men",            "Thriller",       2005, 12.99,  5),
    (11, "Norwegian Wood",                    "Literary Fiction",1987,13.99,  6),
    (12, "Kafka on the Shore",                "Magical Realism",2002, 14.99,  6),
    (13, "One Hundred Years of Solitude",     "Magical Realism",1967, 15.99,  7),
    (14, "Love in the Time of Cholera",       "Literary Fiction",1985,13.99,  7),
    (15, "The Remains of the Day",            "Literary Fiction",1989,12.99,  8),
    (16, "Never Let Me Go",                   "Science Fiction",2005, 12.99,  8),
    (17, "Beloved",                           "Historical Fiction",1987,13.99, 9),
    (18, "Song of Solomon",                   "Literary Fiction",1977,11.99,  9),
    (19, "Do Androids Dream of Electric Sheep?","Science Fiction",1968,11.99,10),
    (20, "The Man in the High Castle",        "Alternate History",1962,10.99,10),
]

CUSTOMERS = [
    (1,  "Alice Novak",     "alice@email.com",   "Zagreb",     2020),
    (2,  "Boris Horvat",    "boris@email.com",   "Split",      2019),
    (3,  "Carla Petrić",   "carla@email.com",   "Rijeka",     2021),
    (4,  "David Kovač",    "david@email.com",   "Osijek",     2018),
    (5,  "Eva Marković",   "eva@email.com",     "Zadar",      2022),
    (6,  "Filip Jurić",    "filip@email.com",   "Zagreb",     2020),
    (7,  "Goran Babić",    "goran@email.com",   "Split",      2021),
    (8,  "Helena Šimić",  "helena@email.com",  "Zagreb",     2019),
    (9,  "Ivan Vuković",  "ivan@email.com",    "Varaždin",  2022),
    (10, "Jana Blažević", "jana@email.com",    "Zagreb",     2018),
    (11, "Karlo Tomić",   "karlo@email.com",   "Dubrovnik",  2023),
    (12, "Lana Filipović","lana@email.com",    "Zagreb",     2021),
]

ORDERS = [
    (1,  1,  "2024-01-15", 27.98),
    (2,  2,  "2024-01-22", 16.99),
    (3,  3,  "2024-02-03", 44.97),
    (4,  4,  "2024-02-14", 12.99),
    (5,  5,  "2024-03-01", 29.98),
    (6,  1,  "2024-03-10", 14.99),
    (7,  6,  "2024-03-18", 25.98),
    (8,  7,  "2024-04-02", 39.97),
    (9,  8,  "2024-04-11", 12.99),
    (10, 2,  "2024-04-20", 28.98),
    (11, 9,  "2024-05-05", 13.99),
    (12, 10, "2024-05-15", 42.97),
    (13, 3,  "2024-05-22", 16.99),
    (14, 11, "2024-06-01", 24.98),
    (15, 12, "2024-06-10", 11.99),
    (16, 4,  "2024-06-18", 27.98),
    (17, 5,  "2024-07-04", 15.99),
    (18, 6,  "2024-07-12", 44.97),
    (19, 7,  "2024-08-01", 12.99),
    (20, 8,  "2024-08-15", 29.98),
]

ORDER_ITEMS = [
    # order 1: 1984 + Animal Farm
    (1,  1,  1,  1, 12.99),
    (2,  1,  2,  1,  9.99),
    # order 2: Dune
    (3,  2,  5,  1, 16.99),
    # order 3: HP1 + HP2 + Norwegian Wood
    (4,  3,  3,  1, 14.99),
    (5,  3,  4,  1, 14.99),
    (6,  3, 11,  1, 13.99),
    # order 4: The Road
    (7,  4,  9,  1, 11.99),
    # order 5: Harry Potter x2
    (8,  5,  3,  1, 14.99),
    (9,  5,  4,  1, 14.99),
    # order 6: Harry Potter Chamber
    (10, 6,  4,  1, 14.99),
    # order 7: 1984 + Left Hand
    (11, 7,  1,  1, 12.99),
    (12, 7,  7,  1, 12.99),
    # order 8: Dune + Dune Messiah + Dispossessed
    (13, 8,  5,  1, 16.99),
    (14, 8,  6,  1, 14.99),
    (15, 8,  8,  1, 12.99),
    # order 9: The Road
    (16, 9,  9,  1, 12.99),
    # order 10: Kafka + One Hundred Years
    (17,10, 12,  1, 14.99),
    (18,10, 13,  1, 15.99),
    # order 11: Norwegian Wood
    (19,11, 11,  1, 13.99),
    # order 12: Beloved + Never Let Me Go + Do Androids
    (20,12, 17,  1, 13.99),
    (21,12, 16,  1, 12.99),
    (22,12, 19,  1, 11.99),
    # order 13: Dune
    (23,13,  5,  1, 16.99),
    # order 14: Remains of Day + Man in High Castle
    (24,14, 15,  1, 12.99),
    (25,14, 20,  1, 10.99),
    # order 15: Do Androids
    (26,15, 19,  1, 11.99),
    # order 16: Love in Cholera + 1984
    (27,16, 14,  1, 13.99),
    (28,16,  1,  1, 12.99),
    # order 17: The Dispossessed
    (29,17,  8,  1, 15.99),
    # order 18: Song of Solomon + Beloved + Kafka
    (30,18, 18,  1, 11.99),
    (31,18, 17,  1, 14.99),
    (32,18, 12,  1, 14.99),
    # order 19: The Road
    (33,19,  9,  1, 12.99),
    # order 20: HP1 + HP2
    (34,20,  3,  1, 14.99),
    (35,20,  4,  1, 14.99),
]


def create():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO authors VALUES (?,?,?,?)", AUTHORS)
    conn.executemany("INSERT INTO books VALUES (?,?,?,?,?,?)", BOOKS)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", CUSTOMERS)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", ORDER_ITEMS)
    conn.commit()

    print(f"Created {DB_PATH}")
    for table in ["authors", "books", "customers", "orders", "order_items"]:
        n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n} rows")

    conn.close()


if __name__ == "__main__":
    create()
