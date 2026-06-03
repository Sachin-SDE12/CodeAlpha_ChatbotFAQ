import json
from sentence_transformers import SentenceTransformer, util
import gradio as gr

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('faq.json', 'r') as f:
    faqs = json.load(f)

faq_questions = list(faqs.keys())
faq_embeddings = model.encode(faq_questions)

print("Model loaded! Starting server...")

def chatbot(message, history):
    user_embedding = model.encode(message)
    similarities = util.cos_sim(user_embedding, faq_embeddings)
    best_match_idx = similarities.argmax()
    score = similarities[0][best_match_idx]
    if score > 0.5:
        return faqs[faq_questions[best_match_idx]]
    else:
        return "Sorry, I couldn't find that. Ask about internship, tasks, or certificate."

gr.ChatInterface(fn=chatbot, title="CodeAlpha FAQ Chatbot").launch()