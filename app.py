import subprocess
import sys
import os

def install_required_packages():
    required_packages = ['PyPDF2', 'google-generativeai', 'tqdm']
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Please install it manually.")
            sys.exit(1)

install_required_packages()

import logging
import PyPDF2
import google.generativeai as generativeai
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("insight_generation.log", encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

CONFIG = {
    "API_KEY": os.getenv("GENERATIVE_API_KEY", "############"),
}

def configure_api():
    generativeai.configure(api_key=CONFIG["API_KEY"])
    return generativeai.GenerativeModel('gemini-1.5-flash')

gemini_model = configure_api()

def check_existing_insights(pdf_folder, pdf_name):
    existing_files = os.listdir(pdf_folder)
    return {int(file.split('-')[-1].split('.')[0]) for file in existing_files if file.startswith(pdf_name) and file.endswith('.md')}

def generate_insights(pdf_title, page_number, page_text):
    prompt = f"""
    Generate structured insights from the following text of {pdf_title}, page {page_number}:

    <text>
    {page_text}
    </text>

    Provide concise, standalone insights using the following structure. Use markdown syntax with bullet points for each section:

    <summary>
    • Summarize the key points of this page in 3-5 bullet points.
    </summary>

    <fact_finding>
    • List significant facts from the page. Use original quotes or polished quotes as appropriate.
    • Ensure each fact has enough context to be understood independently.
    </fact_finding>

    <evidence_used>
    • Paraphrase any evidence presented on this page.
    • Each point should provide sufficient context to be understood on its own.
    </evidence_used>

    <conclusions_made>
    • List any conclusions or claims made on this page.
    • Present each conclusion with adequate context for standalone clarity.
    </conclusions_made>

    <sources_used>
    • If any sources are cited or referenced, list them here.
    • Provide enough context to understand the relevance of each source.
    </sources_used>

    Important:
    1. Include only the sections that are relevant to this specific page's content.
    2. Omit any section that would be empty.
    3. Ensure each bullet point is informative and can be understood independently.
    4. Focus on generating insights rather than just summarizing or analyzing.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text if response else ""
    except Exception as e:
        logger.error(f"Error generating insights for page {page_number} of {pdf_title}: {e}")
        return ""

def save_insights(pdf_folder, pdf_name, page_number, insights_text):
    filename = f"{pdf_folder}/{pdf_name}-page-{page_number}.md"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(insights_text)
    except Exception as e:
        logger.error(f"Error saving insights for {pdf_name} page {page_number}: {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_files = [file for file in os.listdir(script_dir) if file.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_folder = os.path.join(script_dir, pdf_name)
        
        Path(pdf_folder).mkdir(exist_ok=True)
        
        pages_with_insights = check_existing_insights(pdf_folder, pdf_name)
        
        pdf_path = os.path.join(script_dir, pdf_file)
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
            
            with tqdm(total=total_pages, desc=f"Processing {pdf_name}", unit="page") as pbar:
                for page_number in range(total_pages):
                    if page_number not in pages_with_insights:
                        page = reader.pages[page_number]
                        page_text = page.extract_text()
                        
                        if page_text:
                            insights_text = generate_insights(pdf_name, page_number, page_text)
                            save_insights(pdf_folder, pdf_name, page_number, insights_text)
                            logger.info(f"Insights for {pdf_name} page {page_number} saved.")
                        else:
                            logger.warning(f"No text found on page {page_number} of {pdf_name}. Skipping.")
                    else:
                        logger.info(f"Page {page_number} of {pdf_name} already processed. Skipping.")
                    
                    pbar.update(1)

if __name__ == "__main__":
    main()
