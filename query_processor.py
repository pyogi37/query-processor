import os
import datetime
import json
import re
import time
from typing import Dict, List, Union
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

@dataclass
class QueryResult:
    entity: str
    parameter: str
    start_date: str
    end_date: str

class QueryProcessor:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.groq_api_key)
        self.history: List[Dict[str, str]] = []
        self.MAX_HISTORY = 6
        
        self.company_variations = {
            "flipkart": ["flipkart", "flpkart", "flkrt"],
            "amazon": ["amazon", "amzn", "amazon.com"],
            "walmart": ["walmart", "wal-mart", "wmt"],
            "target": ["target", "tgt"],
        }
        
        self.metric_variations = {
            "GMV": ["gmv", "gross merchandise value", "total sales", "total value"],
            "profit": ["profit", "net income", "earnings", "net profit"],
            "revenue": ["revenue", "sales", "top line"],
            "margin": ["margin", "profit margin", "gross margin"],
        }

    def normalize_query(self, query: str) -> str:
        """Normalize query text by standardizing company and metric names."""
        try:
            normalized = query.lower()
            
            # Company normalization
            for company, variations in self.company_variations.items():
                for variant in variations:
                    normalized = re.sub(
                        rf"\b{variant}\b",
                        company,
                        normalized,
                        flags=re.IGNORECASE
                    )
            
            # Metric normalization
            for metric, variations in self.metric_variations.items():
                for variant in variations:
                    normalized = re.sub(
                        rf"\b{variant}\b",
                        metric,
                        normalized,
                        flags=re.IGNORECASE
                    )
            
            return normalized
        except Exception as e:
            print(f"Error in query normalization: {e}")
            return query

    def extract_json_from_text(self, text: str) -> str:
        """Extract JSON array from mixed text response."""
        # Look for JSON array pattern
        array_match = re.search(r'\[\s*{.*}\s*\]', text, re.DOTALL)
        if array_match:
            return array_match.group(0)
        
        # Look for single JSON object pattern
        object_match = re.search(r'{\s*".*}\s*', text, re.DOTALL)
        if object_match:
            return f'[{object_match.group(0)}]'
            
        return text
    
    def get_today_date(self) -> str:
        """Returns today's date in YYYY-MM-DD format"""
        return datetime.today().strftime('%Y-%m-%d')

    def extract_information(self, query: str) -> Union[Dict, List[Dict]]:
        """Extract structured information from the query using Groq API."""
        normalized_query = self.normalize_query(query)
        
        self.history.append({"role": "user", "content": normalized_query})
        self.history = self.history[-self.MAX_HISTORY:]
        
        today_date = self.get_today_date()
    
        system_prompt = f"""
        You are an AI that extracts structured data from user queries. Please respond ONLY with a JSON array containing objects for each entity-parameter combination, in the following format:

        [
         {{
         "entity": "Company Name",
         "parameter": "Performance Metric",
         "start_date": "YYYY-MM-DD",
         "end_date": "YYYY-MM-DD"
         }}
     ]

     Please convert relative dates mentioned in the query into actual start and end dates based on today's date ({today_date}). If no date range is provided, assume the trailing 12 months (last one year plus 1 day) as the default date range.
     Date handling instructions:
     - "last year" refers to the previous full calendar year.
        - "last quarter" refers to the last completed fiscal quarter.
    - "last 6 months" refers to the previous 6 months from today.

    Please convert relative dates mentioned in the query into actual start and end dates based on today's date. If no date range is provided, assume the trailing 12 months as the default date range.
    If the query contains multiple entities or metrics, respond with separate JSON objects for each entity.
    """
    
        # Add role to system prompt
        system_message = {"role": "system", "content": system_prompt}
        user_message = {"role": "user", "content": normalized_query}
    
        # Build the list of messages
        messages = [system_message] + self.history + [user_message]
    
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    messages=messages,
                    model="llama-3.3-70b-versatile",
                    temperature=0.1
                )
            
                result = completion.choices[0].message.content
                # Clean the response and extract JSON
                result = result.replace("```json", "").replace("```", "").strip()
                json_str = self.extract_json_from_text(result)
            
                parsed_result = json.loads(json_str)
                if isinstance(parsed_result, dict):
                    return [parsed_result]
                return parsed_result
            

            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1} failed to parse: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return [{"error": "Failed to parse response"}]
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return [{"error": f"Failed after {MAX_RETRIES} attempts"}]
            
            time.sleep(2 ** attempt)

    def process_query(self, query: str) -> str:
        """Process the query and return formatted JSON response."""
        try:
            result = self.extract_information(query)
            
            if not result or "error" in result[0]:
                return json.dumps({"error": "Invalid query or processing error"})
                
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Error processing query: {str(e)}"})

def main():
    processor = QueryProcessor()
    
    print("Welcome to the Query Processing System!")
    print("Type 'exit' to quit, 'help' for command list")
    
    while True:
        try:
            user_input = input("\nEnter your query: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
                
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- exit: Quit the program")
                print("- help: Show this help message")
                print("\nQuery examples:")
                print("- 'Show Flipkart GMV for last year'")
                print("- 'Compare Amazon and Walmart profit'")
                continue
                
            if not user_input:
                print("Please enter a valid query")
                continue
                
            result = processor.process_query(user_input)
            print("\nResult:")
            print(result)
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            break
            
        except Exception as e:
            print(f"\nError processing query: {str(e)}")

if __name__ == "__main__":
    main()
