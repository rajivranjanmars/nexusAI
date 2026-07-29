import asyncio
from llm.classifier import classify_intent

async def main():
    queries = [
        "What is my current GPA?",
        "I need help registering for Biology 101.",
        "Hello, how are you today?"
    ]
    
    for q in queries:
        print(f"Testing: '{q}'")
        intent, usage = await classify_intent(q)
        print(f"Result:  {intent}  (tokens: {usage.get('total_tokens', 0)})\n")

if __name__ == "__main__":
    asyncio.run(main())
