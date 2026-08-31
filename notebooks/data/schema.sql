-- Taco Alley operational database (demo)
-- SQLite dialect. Dates are stored as ISO-8601 TEXT ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS').

CREATE TABLE stores (
    store_id      INTEGER PRIMARY KEY,
    store_name    TEXT    NOT NULL UNIQUE,   -- e.g. 'Springfield'
    city          TEXT    NOT NULL,
    state         TEXT    NOT NULL,          -- 'PA', 'NJ', 'DE'
    region        TEXT    NOT NULL,          -- 'Northeast', 'Mid-Atlantic'
    opened_date   TEXT    NOT NULL,
    seats         INTEGER NOT NULL,
    has_drive_thru INTEGER NOT NULL          -- 0 / 1
);

CREATE TABLE employees (
    employee_id   INTEGER PRIMARY KEY,
    store_id      INTEGER NOT NULL REFERENCES stores(store_id),
    first_name    TEXT    NOT NULL,
    last_name     TEXT    NOT NULL,
    role          TEXT    NOT NULL,          -- 'Crew', 'Shift Lead', 'Assistant Manager', 'General Manager'
    hire_date     TEXT    NOT NULL,
    hourly_rate   REAL    NOT NULL,
    is_active     INTEGER NOT NULL           -- 0 / 1
);

CREATE TABLE menu_items (
    item_id        INTEGER PRIMARY KEY,
    item_name      TEXT    NOT NULL UNIQUE,
    category       TEXT    NOT NULL,         -- 'Taco', 'Burrito', 'Bowl', 'Side', 'Drink', 'Dessert'
    price          REAL    NOT NULL,
    food_cost      REAL    NOT NULL,
    is_limited_time INTEGER NOT NULL,        -- 0 / 1
    introduced_date TEXT   NOT NULL,
    is_active      INTEGER NOT NULL          -- 0 / 1
);

CREATE TABLE loyalty_members (
    member_id     INTEGER PRIMARY KEY,
    home_store_id INTEGER NOT NULL REFERENCES stores(store_id),
    join_date     TEXT    NOT NULL,
    tier          TEXT    NOT NULL,          -- 'Bronze', 'Silver', 'Gold'
    points_balance INTEGER NOT NULL
);

CREATE TABLE orders (
    order_id       INTEGER PRIMARY KEY,
    store_id       INTEGER NOT NULL REFERENCES stores(store_id),
    employee_id    INTEGER REFERENCES employees(employee_id),
    member_id      INTEGER REFERENCES loyalty_members(member_id),  -- NULL for guest orders
    channel        TEXT    NOT NULL,         -- 'In-Store', 'Drive-Thru', 'Mobile App', 'Delivery'
    order_ts       TEXT    NOT NULL,         -- 'YYYY-MM-DD HH:MM:SS'
    subtotal       REAL    NOT NULL,
    discount       REAL    NOT NULL,
    tax            REAL    NOT NULL,
    tip            REAL    NOT NULL,
    total          REAL    NOT NULL,
    payment_method TEXT    NOT NULL,         -- 'Card', 'Cash', 'Mobile Wallet', 'Gift Card'
    status         TEXT    NOT NULL          -- 'Completed', 'Refunded', 'Cancelled'
);

CREATE TABLE order_items (
    order_line_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    item_id       INTEGER NOT NULL REFERENCES menu_items(item_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL    NOT NULL,
    line_total    REAL    NOT NULL
);

CREATE TABLE customer_tickets (
    ticket_id     INTEGER PRIMARY KEY,
    store_id      INTEGER NOT NULL REFERENCES stores(store_id),
    store_name    TEXT    NOT NULL,          -- denormalized copy of stores.store_name
    order_id      INTEGER REFERENCES orders(order_id),
    channel       TEXT    NOT NULL,          -- 'Phone', 'Email', 'App', 'In-Person', 'Social'
    category      TEXT    NOT NULL,          -- 'Complaint', 'Compliment', 'Question', 'Refund Request'
    subject       TEXT    NOT NULL,
    priority      TEXT    NOT NULL,          -- 'Low', 'Medium', 'High'
    status        TEXT    NOT NULL,          -- 'Open', 'In Progress', 'Resolved', 'Closed'
    created_at    TEXT    NOT NULL,          -- 'YYYY-MM-DD HH:MM:SS'
    resolved_at   TEXT,                      -- NULL when unresolved
    satisfaction_score INTEGER               -- 1-5, NULL when not surveyed
);

CREATE INDEX idx_orders_store_ts ON orders(store_id, order_ts);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_item ON order_items(item_id);
CREATE INDEX idx_tickets_store_created ON customer_tickets(store_id, created_at);
CREATE INDEX idx_tickets_category ON customer_tickets(category);
