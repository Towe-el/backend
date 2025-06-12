import asyncio
from test_conversation_guide import test_conversation_guide_service, test_cumulative_input
 
if __name__ == "__main__":
    print("Running Conversation Guide Service Tests...")
    asyncio.run(test_conversation_guide_service())
    print("\n" + "="*60)
    asyncio.run(test_cumulative_input()) 