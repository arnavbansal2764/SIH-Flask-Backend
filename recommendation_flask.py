from flask import Flask, request, jsonify
import requests
import fitz  # PyMuPDF
import os
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

def download_pdf(pdf_url, pdf_path):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(pdf_path, 'wb') as file:
            file.write(response.content)
    else:
        raise ValueError("Failed to download PDF file.")

def pdf_to_text(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF file: {str(e)}")
    
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    
    doc.close()
    return text

def load_data(pdf_path, n):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()

    list1 = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(data)

    for document in documents:
        data_text = document.page_content
        data_text = data_text.replace("\n", "")
        list1.append(data_text)

    for i, data in enumerate(list1):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=data)
        embedding = response["embedding"]
        collection.add(ids=[str(i) + str(n)], embeddings=[embedding], documents=[data])

    print("Loaded")

def get_bot_response(user_input):
    prompt = user_input
    response = ollama.embeddings(prompt=prompt, model="mxbai-embed-large")
    results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
    if results["documents"]:
        data = results["documents"][0][0]
        output = ollama.generate(
            model="wizardlm2",
            prompt=f"Using this data: {data}. Respond to this prompt: {prompt}",
        )
        return output["response"]
    else:
        return "No relevant data found."

@app.route('/recommendation', methods=['POST'])
def recommendation():
    data = request.json
    pdf_url = data.get('pdf_url')
    job_description = data.get('job_description')
    
    if not pdf_url or not job_description:
        return jsonify({"error": "Missing pdf_url or job_description"}), 400
    
    pdf_path = "resume.pdf"
    try:
        download_pdf(pdf_url, pdf_path)
        load_data(pdf_path, "1")
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    qu = f"""
    Job Description:

        You will critically and in a high level analyze a job seeker's profile based on the following job description and evaluate it across several key parameters. Provide an overall rating out of 100, considering the alignment of the job seeker’s profile which is given to you with the job description. The parameters you should consider are: Skills Match, Experience, Education, Accomplishments, Cultural Fit, Geographical Fit, Career Progression, Availability, Industry Knowledge, and Recommendations/References.

    ---

    Job Description:
    {job_description}

    ---

    Instructions:

    1. Skills Match (0-20 points):
       - Evaluate how well the job seeker's technical and soft skills align with the job requirements.

    2. Experience (0-20 points):
       - Consider the relevance of the job seeker's years of experience, prior roles, and industry experience to the job description.

    3. Education (0-10 points):
       - Assess the relevance of the job seeker’s educational background, including degrees and certifications, to the job requirements.

    4. Accomplishments (0-10 points):
       - Evaluate the significance of the job seeker's professional achievements and project experience in relation to the job description.

    5. Cultural Fit (0-10 points):
       - Determine how well the job seeker’s values and adaptability align with the company’s culture and work environment.

    6. Geographical Fit (0-5 points):
       - Consider the job seeker’s location in relation to the job location and their willingness to relocate, if necessary.

    7. Career Progression (0-10 points):
       - Evaluate the job seeker’s career growth trajectory and their ability to take on increasing responsibilities.

    8. Availability (0-5 points):
       - Assess the job seeker’s availability to start work and their willingness to commit to the company.

    9. Industry Knowledge (0-5 points):
       - Evaluate the job seeker’s understanding of industry trends and market conditions, and their ability to contribute valuable insights.

    10. Recommendations/References (0-5 points):
        - Assess the quality and relevance of the job seeker’s recommendations or references.

    ---

    Output format:

    Overall Rating: [Sum of all points out of 100]

    Strengths:
    - [Strength 1]
    - [Strength 2]
    - ...

    Weaknesses:
    - [Weakness 1]
    - [Weakness 2]
    - ...

    Overall Assessment:
    [Provide a brief summary of the job seeker’s strengths and weaknesses in relation to the job description, highlighting the key factors that influenced the overall rating.]
    """
    
    response = get_bot_response(qu)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
