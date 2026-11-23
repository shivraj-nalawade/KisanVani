import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=15.0
)

def generate_response(query: str, context: str, max_retries: int = 3) -> str:
    print("Generating answer with Groq (Model: openai/gpt-oss-120b)...")
    
    prompt = f"""
    You are an expert agricultural assistant named KisanVani, helping Indian farmers.
    
    RULES:
    1. Answer the farmer's question using ONLY the provided Context below.
    2. Keep the answer very short, simple, and spoken-style (1-3 sentences maximum).
    3. IMPORTANT: You must reply in the EXACT SAME LANGUAGE the farmer used in their question.
    4. If the answer is not contained in the Context, politely say that you do not have that information.

    Context from Agricultural Documents:
    {context}
    """
    
    for attempt in range(max_retries):
        try:
            # Using your specific model and playground parameters
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                temperature=1,
                max_completion_tokens=2048,
                top_p=1,
                reasoning_effort="medium",
                stream=False,
                stop=None
            )
            
            answer = completion.choices[0].message.content.strip()
            print(f"\n[LLM ANSWER]: {answer}\n")
            return answer
            
        except Exception as e:
            print(f"\n[LLM WARNING]: Attempt {attempt + 1}/{max_retries} failed. Error: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("\n[LLM ERROR]: Max retries reached.\n")
                return "I am currently experiencing network delays. Please try again."