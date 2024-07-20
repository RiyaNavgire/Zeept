import requests


# # Base URL of the API
# BASE_URL = "http://128.199.24.128:11434/api/generate"

# # Headers including the authorization token
# headers = {
#      "Content-Type": "application/json"
# }

# url = f"{BASE_URL}"
    
# data = {
#         "model": "llama2",
#         "prompt": "How many colors in rainbow"
        
#     }
    
# response = requests.post(url,json=data)
    
# if response.status_code == 200:
#    print(response.json())





#***************
from langchain.embeddings import HuggingFaceInstructEmbeddings
import requests
from langchain.llms.base import LLM
from typing import Optional, List

class CustomLLM(LLM):
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": "llama2",  # Adjust model name if necessary
            "prompt": prompt
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if "context" in result:
            tokens = result["context"]
            return self.decode_tokens(tokens)
        else:
            raise ValueError("Unexpected response format")

    def decode_tokens(self, tokens: List[int]) -> str:
        
        embeddings = HuggingFaceInstructEmbeddings(
            #api_key=HF_key, 
            model_name="hkunlp/instructor-base",  #https://huggingface.co/hkunlp
            #HuggingFaceInferenceAPIEmbeddings #model_name="sentence-transformers/all-MiniLM-L6-v2",
            
        )
        return embeddings.decode(tokens, skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "custom_llm"