

# Flask Application API Documentation

## Overview

This Flask application provides endpoints for processing PDFs, analyzing job descriptions, and generating resume recommendations. Below is a detailed description of each route, including request methods and example requests.

## Endpoints

### 1. `/recommendation`

#### Method: `POST`

**Description:** 
This endpoint processes a PDF file from a URL and a job description. It converts the PDF to text, processes the text to create embeddings, and then evaluates a job seeker's profile based on the provided job description.

**Request JSON:**
```json
{
  "pdf_url": "https://utfs.io/f/uYCJGxAcJId8uk6is3AcJId8F3SNyhRlvDtHqWgfLOrVjk70",
  "job_description": "As an AI intern at Wastelink, you'll have the opportunity to work on cutting-edge solutions. Your role will involve using your knowledge of artificial intelligence and machine learning to develop innovative tools and algorithms that will contribute to our sustainability efforts. You will work on designing and implementing computer vision algorithms to detect and identify objects and extract specific data from them. Responsibilities include building a prototype, data annotation, model training using frameworks like TensorFlow or PyTorch, and integration with warehouse management systems. This is an excellent opportunity to gain hands-on experience in AI and machine learning, contributing to the automation and efficiency of our warehousing processes. If you are a passionate and driven individual with a strong background in AI and machine learning, this internship at Wastelink is the perfect opportunity to gain hands-on experience and make a real impact in the field of sustainable waste management. Apply now and join us in shaping a cleaner and greener future! About Company: Wastelink is a food surplus management company that helps food manufacturers manage their surplus and waste by transforming it into nutritional feeds for animals. Our mission is to supercharge the circular economy and eliminate food waste. We process thousands of tons of food surplus into high-energy feed ingredients trusted by the world's leading feed brands while providing food manufacturers with a truly sustainable way of managing their waste. Desired Skills and Experience Machine Learning, Artificial Intelligence, Data Science, Deep Learning, Data Structures"
}

```

**Response JSON:**
```json
{
  "response": "Generated recommendation based on the provided job description and resume."
}
```

### 2. `/interview`

#### Method: `GET`

**Description:** 
This endpoint opens a page where clicking a button starts recording audio for further emotion and speech analysis.

**Example Request:**
```plaintext
GET /interview
```

**Response:**
A web page with a button to start recording audio.

## Error Handling

- **400 Bad Request:** If required parameters (`pdf_url` or `job_description`) are missing in the request.
- **404 Not Found:** If the requested PDF URL is not accessible.
- **500 Internal Server Error:** For any other errors encountered during processing.

## Setup Instructions

1. **Install Dependencies:**

   ```bash
   pip install Flask requests pymupdf langchain_text_splitters chromadb ollama langchain_community
   ```

2. **Run the Application:**

   ```bash
   python recommendation_flask.py
   ```

3. **Test the Endpoints:**

   Use a tool like Postman or `curl` to send requests to the endpoints described above.

### 3. `/analyse-resume`

#### Method: `GET`

**Request JSON:**
```json
{
  "pdf_url": "https://utfs.io/f/uYCJGxAcJId8uk6is3AcJId8F3SNyhRlvDtHqWgfLOrVjk70"
}

```

**Response JSON:**
```json
{
  "response": " "
}
```
### 4. `/similarity-score`

#### Method: `POST`

**Request JSON:**
```json
{
  "job_description": "Job description text here",
  "resume_pdf_url": "https://utfs.io/f/uYCJGxAcJId8uk6is3AcJId8F3SNyhRlvDtHqWgfLOrVjk70"
}

```

**Response JSON:**
```json
{
  "entity_match_score": 0.0,
  "final_comprehensive_score": 0.13513469696044922,
  "keyword_match_score": 0.0,
  "semantic_similarity": 0.27026939392089844
}
```
