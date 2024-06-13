import openai
from keybert import KeyBERT
import requests
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import time

# Set your OpenAI API key
openai.api_key = "sk..."

# Define a custom LLM class to use with KeyBERT
class OpenAIWrapper:
    def __init__(self, client):
        self.client = client

    def extract_keywords(self, docs, candidates=None):
        prompt = f"Extract keywords from the following document:\n{docs[0]}"
        response = self.client.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response['choices'][0]['text'].strip().split('\n')

# Create your LLM with the OpenAI client
llm = OpenAIWrapper(client=openai)

# Load it in KeyBERT
print("Initializing KeyBERT model...")
kw_model = KeyBERT(model=llm)

# Function to get text from URL with retry mechanism
def get_text_from_url(url):
    try:
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        
        print(f"Fetching URL: {url}")
        response = session.get(url, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        print("URL fetched successfully")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from <p> tags (common in web articles)
        paragraphs = soup.find_all('p')
        document_text = ' '.join([para.get_text() for para in paragraphs])
        return document_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

def extract_keywords_with_retry(kw_model, document_text, retries=5, backoff_factor=1):
    for attempt in range(retries):
        try:
            print(f"Extracting keywords (attempt {attempt + 1})...")
            keywords = kw_model.extract_keywords([document_text])
            return keywords
        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {e}")
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Exiting.")
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

# URL of the document
url = "https://money.usnews.com/money/retirement/aging/articles/common-scams-that-target-seniors-and-how-to-avoid-them"

# Get the document text
print("Fetching document text from URL...")
document_text = get_text_from_url(url)

if document_text:
    print("Document text successfully retrieved. Extracting keywords...")
    try:
        # Extract keywords with retry mechanism
        keywords = extract_keywords_with_retry(kw_model, document_text)
        print("Keywords extracted:")
        print(keywords)
    except Exception as e:
        print(f"Failed to extract keywords: {e}")
else:
    print("Failed to retrieve document text from the URL.")
