CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    op_number TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    age INTEGER,
    sex TEXT,
    record_date TEXT,
    created_at TEXT
);
