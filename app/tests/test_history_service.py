import pytest
import asyncio
from app.services.history_service import HistoryService


class TestHistoryService:
    """Test cases for HistoryService title generation"""

    @pytest.fixture
    def history_service(self):
        """Create a HistoryService instance for testing"""
        return HistoryService()

    @pytest.mark.asyncio
    async def test_generate_title_with_emotional_content(self, history_service):
        """Test title generation with emotional content"""
        accumulated_text = """
        I've been feeling really overwhelmed lately with everything going on at work. 
        My manager keeps piling more tasks on me and I don't know how to handle it all. 
        I feel like I'm drowning and can't catch up. The stress is affecting my sleep 
        and I'm constantly anxious about deadlines. I don't want to disappoint anyone 
        but I'm reaching my breaking point.
        """
        
        title = await history_service.generate_title(accumulated_text)
        
        assert title is not None
        assert isinstance(title, str)
        assert len(title.split()) <= 10
        assert len(title) > 0
        print(f"Generated title: '{title}'")

    @pytest.mark.asyncio
    async def test_generate_title_with_relationship_issue(self, history_service):
        """Test title generation with relationship content"""
        accumulated_text = """
        My best friend and I had a huge fight last week and we haven't spoken since. 
        It started over something so small but it escalated quickly. I said some things 
        I regret and I think she did too. I miss her so much but I don't know how to 
        reach out without making things worse. I keep replaying the conversation in my head 
        and wondering if our friendship is over.
        """
        
        title = await history_service.generate_title(accumulated_text)
        
        assert title is not None
        assert isinstance(title, str)
        assert len(title.split()) <= 10
        print(f"Generated title: '{title}'")

    @pytest.mark.asyncio
    async def test_generate_title_empty_input(self, history_service):
        """Test title generation with empty input"""
        title = await history_service.generate_title("")
        assert title is None

        title = await history_service.generate_title("   ")
        assert title is None

    @pytest.mark.asyncio
    async def test_generate_title_short_input(self, history_service):
        """Test title generation with very short input"""
        short_text = "I feel sad."
        title = await history_service.generate_title(short_text)
        
        # Should still generate a title even for short input
        assert title is not None
        assert isinstance(title, str)
        assert len(title.split()) <= 10
        print(f"Generated title for short input: '{title}'")


if __name__ == "__main__":
    # Simple test runner for development
    async def run_tests():
        service = HistoryService()
        
        test_cases = [
            "I've been feeling really overwhelmed with work and can't manage all the stress anymore.",
            "My relationship is falling apart and I don't know what to do about it.",
            "I'm struggling with anxiety and depression after losing my job last month.",
            "Family conflicts are tearing us apart and I feel caught in the middle."
        ]
        
        for i, text in enumerate(test_cases, 1):
            print(f"\nTest case {i}:")
            print(f"Input: {text}")
            title = await service.generate_title(text)
            print(f"Generated title: '{title}'")
            print(f"Word count: {len(title.split()) if title else 0}")
    
    # Run the tests if this file is executed directly
    asyncio.run(run_tests()) 