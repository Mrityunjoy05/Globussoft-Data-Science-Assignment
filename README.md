# Globussoft Data Science Assignment

A Python-based project covering two tasks — Amazon product scraping and face verification via a FastAPI service.

---

## Project Structure

```
globussoft/
│
├── task1/
│   ├── amazon_scraper.py
│   └── amazon_laptops_YYYYMMDD_HHMMSS.csv
│
├── task2/
│   ├── train.py
│   ├── predict.py
│   └── app.py
│
├── sample_images/
│   ├── person1_photo1.jpg     # same person — photo 1
│   ├── person1_photo2.jpg     # same person — photo 2 
│   ├── person2.jpg            # different person A
│   └── person3.jpg            # different person B
│
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- No GPU required — runs on CPU

```bash
pip install -r requirements.txt
```

or with uv:

```bash
uv add -r requirements.txt
```

---

## Task 1 — Amazon Laptop Scraper

Scrapes laptop listings from `amazon.in` and saves results to a timestamped CSV file.

**Fields collected:** Title, Image URL, Rating, Price, Result Type (Ad / Organic)

```bash
cd task1
python amazon_scraper.py
```

Output: `amazon_laptops_YYYYMMDD_HHMMSS.csv` saved in the `task1/` folder.

> The scraper waits 2–5 seconds between page turns to mimic human browsing behaviour.

---

## Task 2 — Face Authentication (Face Verification)

A FastAPI service that accepts two face images and determines whether they belong to the same person.

**Model:** InsightFace `buffalo_l` — ArcFace pretrained on MS1MV3 (5 million face images). Weights download automatically on first run.

**How it works:**

```
Image 1 ──► Face Detection ──► ArcFace Embedding (512-d)
                                         │
                                Cosine Similarity
                                         │
Image 2 ──► Face Detection ──► ArcFace Embedding (512-d)
                                         │
                            score ≥ 0.55 → "same person"
                            score < 0.55 → "different person"
```

---

### Step 1 — Prepare the model

```bash
cd task2
python train.py
```

Downloads the pretrained model on first run (~1.7 GB) and saves `model/config.json`.

```

---

### Step 2 — Test prediction from terminal

```bash
python predict.py ../sample_images/person1_photo1.jpg ../sample_images/person1_photo2.jpg
```

Expected output:

```
  verification_result: same person
  similarity_score: 0.7243
  bounding_box_image1: [120, 45, 380, 310]
  bounding_box_image2: [115, 50, 375, 305]
  error: None
```

---

### Step 3 — Start the API server

```bash
fastapi dev app.py
```

Server starts at: `http://127.0.0.1:8000`

---

### Step 4 — Interactive docs

Open in browser:

```
http://127.0.0.1:8000/docs
```

Upload two face images and test the endpoint directly from the browser.

**Response format:**

```json
{
  "verification_result": "same person",
  "similarity_score": 0.7243,
  "bounding_box_image1": [120, 45, 380, 310],
  "bounding_box_image2": [115, 50, 375, 305]
}
```

---

## Notes

- The scraper collects 5 pages by default. Change `pages=5` in `scrape_laptops()` to scrape more.
- The `FutureWarning` about `estimate` printed in the terminal is from InsightFace internally — it does not affect results and can be ignored.