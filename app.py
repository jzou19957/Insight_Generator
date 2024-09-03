import os
import logging
import PyPDF2
import google.generativeai as generativeai
from tqdm import tqdm  # For progress bar
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("data_analysis.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load configuration
CONFIG = {
    "API_KEY": os.getenv("GENERATIVE_API_KEY", "####"),  # Replace with your Google Gemini API key
}

# Configure Google Gemini API
def configure_api():
    generativeai.configure(api_key=CONFIG["API_KEY"])
    return generativeai.GenerativeModel('gemini-1.5-flash')

# Initialize the Gemini API
gemini_model = configure_api()

# Function to check for existing summaries
def check_existing_summaries(pdf_folder):
    """Check which pages already have summaries to avoid redundant processing."""
    existing_files = os.listdir(pdf_folder)
    pages_with_summaries = [
        int(file.split('-')[1].split('.')[0]) 
        for file in existing_files if file.endswith('.md')
    ]
    return pages_with_summaries

# Function to generate a summary using the Gemini model
def summarize_page(pdf_title, page_number, page_text):
    """Generate a summary for a given page using the Gemini model."""
    prompt = f"""
    <pdf_title> {pdf_title} </pdf_title>
    <page> {page_number} </page>
    <text> {page_text} </text>
    
    <summary> Generate a summary of the above text. </summary>
    <fact_finding> List original quotes that are factual. </fact_finding>
    <claims_made> Identify any assertions or claims. </claims_made>
    <evidence_use> Note any evidence presented. </evidence_use>
    <data_statistic_used> Highlight any data or statistics mentioned. </data_statistic_used>
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text if response else ""
    except Exception as e:
        logger.error(f"Error generating summary for page {page_number} of {pdf_title}: {e}")
        return ""

# Function to save summary as a markdown file
def save_summary(pdf_folder, page_number, summary_text):
    """Save the summary as a markdown file."""
    filename = f"{pdf_folder}/page-{page_number}.md"
    with open(filename, 'w') as f:
        f.write(summary_text)

# Main function to read PDF and generate summaries
def main():
    # Get all PDF files in the current directory
    pdf_files = [file for file in os.listdir('.') if file.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_title = os.path.splitext(pdf_file)[0]
        pdf_folder = f"./{pdf_title}"
        
        # Create a folder for each PDF if it doesn't exist
        Path(pdf_folder).mkdir(exist_ok=True)
        
        pages_with_summaries = check_existing_summaries(pdf_folder)
        
        with open(pdf_file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
            
            # Progress bar setup
            with tqdm(total=total_pages, desc=f"Processing {pdf_title}", unit="page") as pbar:
                for page_number in range(total_pages):
                    if page_number not in pages_with_summaries:
                        page = reader.pages[page_number]
                        page_text = page.extract_text()
                        
                        if page_text:
                            # Generate summary for this page
                            summary_text = summarize_page(pdf_title, page_number, page_text)
                            
                            # Save the summary to a markdown file
                            save_summary(pdf_folder, page_number, summary_text)
                            logger.info(f"Summary for {pdf_title} page {page_number} saved.")
                        else:
                            logger.warning(f"No text found on page {page_number} of {pdf_title}. Skipping.")
                    else:
                        logger.info(f"Page {page_number} of {pdf_title} already processed. Skipping.")
                    
                    pbar.update(1)  # Update progress bar

if __name__ == "__main__":
    main()
