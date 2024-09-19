from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import fitz
import requests
app = Flask(__name__)

# Initialize model and NLP
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return set(keywords)

def calculate_keyword_match_score(job_keywords, resume_keywords):
    intersection = len(job_keywords.intersection(resume_keywords))
    union = len(job_keywords.union(resume_keywords))
    return intersection / union if union != 0 else 0

def extract_entities(text):
    doc = nlp(text)
    entities = {ent.text.lower() for ent in doc.ents if ent.label_ in ['ORG', 'GPE', 'PERSON', 'WORK_OF_ART', 'DATE', 'MONEY']}
    return entities

def calculate_entity_match_score(job_entities, resume_entities):
    intersection = len(job_entities.intersection(resume_entities))
    union = len(job_entities.union(resume_entities))
    return intersection / union if union != 0 else 0

def calculate_similarity_score(job_description, resume_text):
    # 1. Semantic Similarity using Sentence-BERT
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    semantic_similarity = util.cos_sim(job_embedding, resume_embedding).item()
    
    # 2. Keyword Matching using TF-IDF
    job_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)
    keyword_match_score = calculate_keyword_match_score(job_keywords, resume_keywords)
    
    # 3. Named Entity Matching using NER
    job_entities = extract_entities(job_description)
    resume_entities = extract_entities(resume_text)
    entity_match_score = calculate_entity_match_score(job_entities, resume_entities)

    weight_semantic = 0.5
    weight_keyword = 0.3
    weight_entity = 0.2
    
    final_score = (weight_semantic * semantic_similarity +
                   weight_keyword * keyword_match_score +
                   weight_entity * entity_match_score)
    
    return {
        'semantic_similarity': semantic_similarity,
        'keyword_match_score': keyword_match_score,
        'entity_match_score': entity_match_score,
        'final_comprehensive_score': final_score
    }

@app.route('/similarity-score', methods=['POST'])
def similarity_score():
    data = request.json
    job_description = data.get('job_description')
    resume_pdf_url = data.get('resume_pdf_url')
    
    if not job_description or not resume_pdf_url:
        return jsonify({"error": "Job description and resume PDF URL are required"}), 400

    # Download PDF and extract text
    try:
        # Download and save the PDF
        response = requests.get(resume_pdf_url)
        pdf_path = 'resume.pdf'
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        
        # Extract text from PDF
        resume_text = pdf_to_text(pdf_path)
        
        # Calculate similarity scores
        scores = calculate_similarity_score(job_description, resume_text)
        return jsonify(scores)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
