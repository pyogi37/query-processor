# Query Processing System Setup and Running Instructions
**https://github.com/pyogi37/query-processor**

## Overview
*This application allows you to extract company performance metrics (such as GMV, Profit, Revenue) from user queries using Groq API. The application processes user queries and returns structured data in JSON format.*

## Prerequisites

Before you begin, ensure you have the following:

1. **Python 3.7+** installed on your system.
2. **Groq API key**: You need an API key from Groq to interact with their API.

## Step 1: Clone the Repository (if applicable)

```bash
git clone https://github.com/your-repo/query-processing-system.git
cd query-processing-system
```

## Step 2: Create a Virtual Environment (optional, but recommended)
It’s a good practice to use a virtual environment to manage dependencies for your Python project.
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
## Step 3: Install Dependencies
Install the required Python libraries using pip by running the following command:
```
pip install dotenv groq
```
Dependencies:
- groq: To interact with the Groq API for performance metrics.
- python-dotenv: To manage environment variables.
datetime: For handling date and time operations.

## Step 4: Set Up the Environment Variables
Create a .env file in the root of your project directory.
Add the following line to the .env file:
```
GROQ_API_KEY=your_groq_api_key_here
```
*Replace your_groq_api_key_here with your actual **Groq API key**. This will allow the application to access the **Groq API**.*

## Step 5: Running the Application
Once you have set up the environment and installed the dependencies, you can run the application.

Running the Application in the Terminal
In the root directory of your project, run the following command:
```
python query_processor.py
```
**Example Usage:**
The program will prompt you to enter a query. Example queries include:

- *"Show Flipkart GMV for last year"*
- *"Compare Amazon and Walmart profit"*

Once a query is entered, the application will output a structured JSON response containing the extracted data for the specified company and metric.

## Step 6: Interacting with the Application
The program will continuously prompt for queries until you type 'exit' to quit.
To view available commands, type 'help'.
Example output:

```
{
  "entity": "Amazon",
  "parameter": "Profit",
  "start_date": "2023-03-16",
  "end_date": "2024-03-15"
}
```
## Troubleshooting
- *If you encounter issues with the Groq API, make sure your API key is correctly set in the .env file.*
- *If the program returns errors related to dependencies, try reinstalling the packages using pip install -r requirements.txt.*