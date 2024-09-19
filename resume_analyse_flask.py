import os
import requests
from flask import Flask, request, jsonify
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import ollama
from langchain_community.document_loaders import PyMuPDFLoader

app = Flask(__name__)

# Initialize ChromaDB client and collection
client = chromadb.Client()
try:
    collection = client.create_collection(name="docs")
except chromadb.db.base.UniqueConstraintError:
    collection = client.get_collection(name="docs")


# Function to download PDF
def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print("PDF downloaded successfully.")


# Function to load and process the PDF
def load_data(path, n):
    loader = PyMuPDFLoader(path)
    data = loader.load()

    list1 = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    text_splitter.split_documents(data)

    for i in range(len(data)):
        data_text = data[i].page_content
        data_text = data_text.replace("\n", "")
        list1.append(data_text)

    for i, data in enumerate(list1):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=data)
        embedding = response["embedding"]
        collection.add(ids=[str(i) + str(n)], embeddings=[embedding], documents=[data])

    print("Data Loaded")


# Function to get bot response
def get_bot_response(user_input):
    prompt = user_input
    response = ollama.embeddings(prompt=prompt, model="mxbai-embed-large")
    results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
    data = results["documents"][0][0]
    print("Data Given To Ollama ")
    output = ollama.generate(
        model="wizardlm2",
        prompt=f"Using this data: {data}. Respond to this prompt: {prompt}",
    )
    return output["response"]


# Define the Flask route for PDF processing and scoring
@app.route('/calc-score', methods=['GET'])
def calc_score():
    # Get the PDF URL from query params
    pdf_url = request.args.get('pdf_url')
    if not pdf_url:
        return jsonify({"error": "No PDF URL provided"}), 400

    # Define the path to save the downloaded PDF
    save_path = 'downloaded_resume.pdf'

    # Step 1: Download the PDF
    try:
        download_pdf(pdf_url, save_path)
    except Exception as e:
        return jsonify({"error": f"Failed to download PDF: {str(e)}"}), 500

    # Step 2: Load and process the PDF
    try:
        load_data(save_path, "1")
    except Exception as e:
        return jsonify({"error": f"Failed to load PDF data: {str(e)}"}), 500

    # Step 3: Perform job description and resume analysis
    job = """
    Objectives of this role
    Collaborate with product design and engineering teams to develop an understanding of needs
    Research and devise innovative statistical models for data analysis
    Communicate findings to all stakeholders
    Enable smarter business processes by using analytics for meaningful insights
    Keep current with technical and industry developments
    Responsibilities
    Serve as lead data strategist to identify and integrate new datasets that can be leveraged through our product capabilities, and work closely with the engineering team in the development of data products
    Execute analytical experiments to help solve problems across various domains and industries
    Identify relevant data sources and sets to mine for client business needs, and collect large structured and unstructured datasets and variables
    Devise and utilize algorithms and models to mine big-data stores; perform data and error analysis to improve models; clean and validate data for uniformity and accuracy
    Analyze data for trends and patterns, and interpret data with clear objectives in mind
    Implement analytical models in production by collaborating with software developers and machine-learning engineers
    Required skills and qualifications
    Seven or more years of experience in data science
    Proficiency with data mining, mathematics, and statistical analysis
    Advanced experience in pattern recognition and predictive modeling
    Experience with Excel, PowerPoint, Tableau, SQL, and programming languages (ex: Java/Python, SAS)
    Ability to work effectively in a dynamic, research-oriented group that has several concurrent projects
    Preferred skills and qualifications
    Bachelor’s degree (or equivalent) in statistics, applied mathematics, or related discipline
    Two or more years of project management experience
    Professional certification

    """  

    prompt = f"""
    1. Extract and Analyze Features
    Job Description:

    Input: {job}
    Task: Extract and list the critical features of the job description, including required skills, qualifications, experience levels, and responsibilities. Identify and prioritize essential criteria based on their importance for the role.
    Resume:

    Task: Extract and list key features from the resume, including skills, work experience, educational background, and certifications. Normalize and structure the extracted data for consistency.
    2. Feature Representation and Comparison
    Keyword and Skill Extraction:

    Job Description: Normalize and compile a list of essential keywords and skills.
    Resume: Normalize and compile a list of keywords and skills present in the resume.
    Vectorization:

    Convert both the job description features and resume features into vectors using advanced NLP techniques (e.g., TF-IDF, word embeddings like BERT).
    3. Calculate Similarity Scores
    Keyword and Skill Matching:

    Task: Compute similarity scores between the keywords and skills from the job description and those found in the resume. Use cosine similarity or other advanced similarity measures to quantify matches.
    Experience and Qualification Matching:

    Task: Assess how well the candidate's work experience and qualifications align with the job description. Evaluate the relevance and depth of experience using semantic similarity measures.
    4. Scoring and Weighting
    Score Calculation:

    Skill Match Score: Based on the frequency and relevance of matched skills and keywords.
    Experience Match Score: Based on the alignment and duration of past job roles relative to the job description.
    Education and Certification Score: Based on the relevance of educational qualifications and certifications.
    Apply Weights:

    Assign weights to each scoring component according to its importance for the role (e.g., skills might have a higher weight compared to education).
    Composite Score:

    Aggregate the individual scores into a composite score using the assigned weights.
    5. Generate Output
    Detailed Analysis:

    Provide a summary of the extracted features from both the job description and resume.
    Include similarity scores for skills, experience, and qualifications.
    Present individual scores and the final composite score, explaining how each score was derived.
    Ranking:

    Rank the resume based on the composite score relative to other resumes (if applicable).
    If necessary, apply a threshold score to filter out resumes that do not meet the minimum requirements.
    6. Final Output
    Resume Analysis Summary: Detailed analysis of how the resume meets or falls short of the job description criteria.
    Composite Score: The final composite score reflecting the overall match.
    Recommendations: Insights or suggestions for improving the resume based on the job description requirements (optional).
    By following this prompt, the AI system should be able to deliver a thorough and insightful analysis of how well the resume aligns with the job description, providing a clear and actionable evaluation.

    """
    try:
        response = get_bot_response(prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)