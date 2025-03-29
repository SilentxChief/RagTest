import os
from dotenv import load_dotenv
from groq import Groq

def get_available_models(client):
    """Get a list of available models from Groq"""
    try:
        # Common Groq models - using a static list as the Groq API may not have a direct endpoint for listing models
        models = [
            "llama3-8b-8192",
            "llama3-70b-8192"
        ]
        return models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["llama3-8b-8192"]  # Return default model if error occurs

def display_model_options(models):
    """Display available models as a numbered list"""
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    return len(models)

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Get available models
    models = get_available_models(client)
    
    # Display model options
    model_count = display_model_options(models)
    
    # Get user model selection
    while True:
        try:
            model_choice = int(input(f"\nSelect a model (1-{model_count}): "))
            if 1 <= model_choice <= model_count:
                selected_model = models[model_choice - 1]
                print(f"Selected model: {selected_model}")
                break
            else:
                print(f"Please enter a number between 1 and {model_count}")
        except ValueError:
            print("Please enter a valid number")
    
    # Get user question
    user_question = input("\nPlease enter your question: ")
    
    # Query the LLM
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "You are a helpful, concise, and professional assistant."},
                {"role": "user", "content": user_question}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        # Print the response
        print("\nAnswer:")
        print(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
