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
    "API_KEY": os.getenv("GENERATIVE_API_KEY", "###"),
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
    Analyze the following text from {pdf_title}, page {page_number}, and provide structured insights:

    <text>
    {page_text}
    </text>

    Generate a comprehensive analysis using the following structure:

    <key_insights>
    Identify and explain 3-5 crucial points from the text. Critically evaluate their significance and potential implications.
    </key_insights>

    <contextual_analysis>
    Explain how this page's content relates to the broader themes of the document. Consider historical, cultural, or theoretical contexts that enhance understanding.
    </contextual_analysis>

    <evidence_evaluation>
    <facts>List key facts presented, assessing their reliability and relevance.</facts>
    <data>Analyze any quantitative information, discussing its validity and implications.</data>
    <sources>Evaluate the credibility and potential biases of any sources cited.</sources>
    </evidence_evaluation>

    <critical_observations>
    Identify any assumptions, logical flaws, or potential biases in the text. Discuss their impact on the overall argument or presentation.
    </critical_observations>

    <connections_and_patterns>
    Highlight any connections to previous content or potential foreshadowing. Identify emerging patterns or themes.
    </connections_and_patterns>

    <questions_and_implications>
    <questions>Pose 2-3 thought-provoking questions arising from the content.</questions>
    <implications>Discuss potential real-world applications or consequences of the information presented.</implications>
    </questions_and_implications>

    Ensure your analysis is thorough, balanced, and encourages deep contextual thinking.
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
