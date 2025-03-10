import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenRouterAI:
    """
    Class to handle interactions with the OpenRouter API for AI analysis.
    Uses the Deepseek AI model for data analysis.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenRouter AI client."""
        # Use the provided API key or the environment variable
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or "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"

        # Log API key status (not the actual key)
        if self.api_key:
            logger.info("OpenRouter API key loaded")
        else:
            logger.error("No OpenRouter API key available")

        try:
            # Initialize the OpenAI client with OpenRouter base URL
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )

            logger.info("OpenRouter AI client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenRouter AI client: {str(e)}")
            logger.error(traceback.format_exc())
            self.client = None

        # Model configuration
        self.model = "qwen/qwq-32b:free"  # Updated model
        self.http_referer = "https://smartdatahub.com"  # Replace with your actual domain
        self.site_name = "SmartDataHub"    # Your app name

    def analyze_data_initial(self, data_summary: Dict[str, Any]) -> str:
        """
        Generate an initial analysis of uploaded data.

        Args:
            data_summary: A dictionary containing summary information about the uploaded data

        Returns:
            str: The AI's initial impression of the data
        """
        try:
            if not self.client:
                return "AI service is not properly initialized. Please check your configuration."

            logger.info("Generating initial data analysis")
            logger.debug(f"Data summary for analysis: {json.dumps(data_summary, indent=2)}")

            # Create a prompt that instructs the AI to analyze the data summary
            prompt = f"""
            You are an expert data analyst AI assistant for SmartDataHub.

            Analyze the following data summary and provide an initial impression:

            {json.dumps(data_summary, indent=2)}

            Give a concise, insightful initial analysis that highlights:
            1. Key metrics and what they indicate
            2. Notable patterns or trends
            3. Potential areas of interest for deeper analysis

            Keep your response under 200 words and focus on being helpful and insightful.
            """

            response = self.generate_response(prompt)
            logger.info("Initial analysis generated successfully")
            return response

        except Exception as e:
            logger.error(f"Error generating initial analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return "I'm unable to analyze this data at the moment. Please try again later."

    def answer_question(self, question: str, data_context: Dict[str, Any], chat_history: List[Dict[str, str]] = None) -> str:
        """
        Answer a user question about their data.

        Args:
            question: The user's question
            data_context: Context about the data being analyzed
            chat_history: Previous messages in the conversation

        Returns:
            str: The AI's response to the question
        """
        try:
            if not self.client:
                return "AI service is not properly initialized. Please check your configuration."

            logger.info(f"Answering question: {question}")

            # Format the chat history for context
            messages = []

            # Include system message with context
            system_message = f"""
            You are an expert data analyst AI assistant for SmartDataHub.

            DATA CONTEXT:
            {json.dumps(data_context, indent=2)}

            Provide helpful, accurate, and insightful answers based on the data context.
            If you cannot answer based on the available data, explain why and suggest what additional data might help.
            """

            messages.append({"role": "system", "content": system_message})

            # Add chat history if available
            if chat_history:
                for msg in chat_history[-10:]:  # Include last 10 messages for context
                    if msg.get("role") in ["user", "assistant"]:
                        messages.append({
                            "role": msg.get("role"),
                            "content": msg.get("content", "")
                        })

            # Add the current question
            messages.append({"role": "user", "content": question})

            logger.debug(f"Sending {len(messages)} messages to AI")

            # Get response using the chat completion API
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.http_referer,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            response = completion.choices[0].message.content
            logger.info("Successfully received AI response")
            return response

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            logger.error(traceback.format_exc())
            return f"I'm unable to process your question at the moment. Error: {str(e)}"

    def suggest_visualizations(self, data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest appropriate visualizations based on the data.

        Args:
            data_summary: A dictionary containing summary information about the uploaded data

        Returns:
            List[Dict[str, Any]]: List of suggested visualization configs
        """
        try:
            if not self.client:
                return [{"type": "bar", "title": "Default Visualization", "description": "AI service not available"}]

            logger.info("Generating visualization suggestions")

            prompt = f"""
            You are an expert data visualization AI assistant.

            Based on the following data summary, suggest appropriate visualizations:

            {json.dumps(data_summary, indent=2)}

            Return a JSON list of visualization suggestions. Each suggestion should include:
            1. type: The type of chart (e.g., "line", "bar", "pie", "scatter")
            2. title: A descriptive title for the chart
            3. description: Why this visualization would be insightful
            4. data_fields: Which fields from the data should be used

            Only respond with valid JSON. Do not include any other text.
            """

            response = self.generate_response(prompt)
            logger.debug(f"Raw visualization suggestion response: {response}")

            # Clean the response to ensure it's valid JSON
            # Remove any markdown code block markers or extra text
            json_str = response.replace("```json", "").replace("```", "").strip()

            try:
                # Parse the JSON response
                visualization_suggestions = json.loads(json_str)
                logger.info(f"Successfully parsed {len(visualization_suggestions)} visualization suggestions")
                return visualization_suggestions
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to parse AI response as JSON: {response}")
                logger.error(f"JSON decode error: {str(json_error)}")
                # Return a default visualization
                return [{"type": "bar", "title": "Default Visualization", "description": "Basic data overview"}]

        except Exception as e:
            logger.error(f"Error suggesting visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            return [{"type": "bar", "title": "Default Visualization", "description": "Basic data overview"}]

    def generate_response(self, prompt: str) -> str:
        """
        Send a prompt to the OpenRouter API and get a response.

        Args:
            prompt: The text prompt to send to the AI

        Returns:
            str: The AI's response
        """
        try:
            if not self.client:
                return "AI service is not properly initialized."

            logger.debug(f"Sending prompt to AI: {prompt[:100]}...")

            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.http_referer,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )

            response = completion.choices[0].message.content
            logger.debug(f"Received AI response: {response[:100]}...")
            return response

        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            logger.error(traceback.format_exc())
            return f"An error occurred while communicating with the AI service: {str(e)}"

    def create_data_story(self, data_summary: Dict[str, Any]) -> str:
        """
        Create a narrative data story from the provided data summary.

        Args:
            data_summary: A dictionary containing summary information about the uploaded data

        Returns:
            str: A narrative data story explaining key insights
        """
        try:
            if not self.client:
                return "AI service is not properly initialized. Please check your configuration."

            logger.info("Generating data story")

            prompt = f"""
            You are an expert data storyteller for SmartDataHub.

            Based on the following data summary, create a compelling narrative data story:

            {json.dumps(data_summary, indent=2)}

            Your data story should:
            1. Start with a high-level summary of what the data represents
            2. Identify 3-5 key insights from the data
            3. Explain potential business implications
            4. Suggest follow-up questions or analyses

            Write in a clear, engaging style suitable for business stakeholders.
            """

            response = self.generate_response(prompt)
            logger.info("Data story generated successfully")
            return response

        except Exception as e:
            logger.error(f"Error creating data story: {str(e)}")
            logger.error(traceback.format_exc())
            return "I'm unable to create a data story at the moment. Please try again later."

# Initialize the OpenRouterAI instance
def get_ai_instance(api_key=None):
    """
    Get an instance of the OpenRouterAI class.

    Args:
        api_key: Optional API key to use

    Returns:
        OpenRouterAI: An instance of the OpenRouterAI class
    """
    try:
        ai = OpenRouterAI(api_key)
        return ai
    except Exception as e:
        logger.error(f"Error initializing AI instance: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Example usage
if __name__ == "__main__":
    # This will be useful for testing the module independently
    ai = OpenRouterAI()

    # Example data summary for testing
    test_data = {
        "file_name": "sales_data.csv",
        "num_rows": 1500,
        "num_columns": 10,
        "column_names": ["Date", "Product", "Region", "Sales", "Quantity", "Price", "Customer", "Category", "Profit", "Discount"],
        "data_types": {
            "Date": "datetime64[ns]",
            "Product": "object",
            "Region": "object",
            "Sales": "float64",
            "Quantity": "int64",
            "Price": "float64",
            "Customer": "object",
            "Category": "object",
            "Profit": "float64",
            "Discount": "float64"
        },
        "summary_stats": {
            "Sales": {"mean": 1250.45, "min": 100.0, "max": 5000.0},
            "Quantity": {"mean": 25, "min": 1, "max": 100},
            "Profit": {"mean": 300.75, "min": -200.0, "max": 1500.0}
        }
    }

    # Test initial analysis
    print("\nTesting Initial Analysis:")
    initial_analysis = ai.analyze_data_initial(test_data)
    print(initial_analysis)
    print("\n" + "-"*50 + "\n")

    # Test visualization suggestions
    print("Testing Visualization Suggestions:")
    viz_suggestions = ai.suggest_visualizations(test_data)
    print(json.dumps(viz_suggestions, indent=2))
    print("\n" + "-"*50 + "\n")

    # Test question answering
    print("Testing Question Answering:")
    question = "What are the highest and lowest profit values in the data?"
    answer = ai.answer_question(question, test_data)
    print(f"Q: {question}")
    print(f"A: {answer}")