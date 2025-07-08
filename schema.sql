CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    op_number TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    age INTEGER,
    sex TEXT,
    date TEXT,
    user_id TEXT,
    image_filename TEXT
);
