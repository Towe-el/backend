# Toweel Backend Tests

This directory contains comprehensive tests for the Toweel backend application.

## Test Structure

### Core Test Files

1. **`test_conversation_guide.py`** - Tests for the ConversationGuideService
   - Tests conversation quality analysis with different input scenarios
   - Tests cumulative input functionality across multiple rounds
   - Tests guidance response generation and ready-for-search determination
   - Uses mocked Gemini AI client to avoid external API calls

2. **`test_retry_mechanism.py`** - Tests for Vertex AI retry mechanisms
   - Tests retry behavior on ResourceExhausted, ServiceUnavailable, and DeadlineExceeded errors
   - Tests caching behavior during retries using LRU cache
   - Tests error handling for non-retryable errors (ValueError)
   - Tests max retries exceeded scenarios
   - Uses comprehensive mocking of Vertex AI TextEmbeddingModel

3. **`test_rag_service.py`** - Tests for RAG (Retrieval-Augmented Generation) service
   - Tests emotion analysis and filtering based on valence-arousal coordinates
   - Tests quote generation and analysis for different emotions
   - Tests search results processing with emotion statistics
   - Tests error handling for API failures
   - Uses mocked Gemini AI client for content generation

4. **`test_api_endpoints.py`** - Tests for FastAPI endpoints
   - Tests health check endpoint with database status
   - Tests search endpoints with session management
   - Tests search execution with RAG analysis
   - Tests error handling and input validation
   - Uses FastAPI TestClient with comprehensive service mocking

5. **`test_session_service.py`** - Tests for SessionService
   - Tests session creation, retrieval, and updates
   - Tests database unavailable scenarios
   - Tests service initialization with and without database
   - Uses mocked MongoDB async operations

### Test Runners

1. **`run_conversation_guide_test.py`** - Dedicated runner for conversation guide tests
2. **`run_all_tests.py`** - Comprehensive test runner for all tests with summary reporting

## Running Tests

### Run All Tests
```bash
python app/tests/run_all_tests.py
```

### Run Individual Test Suites
```bash
# Conversation Guide tests
python app/tests/run_conversation_guide_test.py

# Retry mechanism tests
python -m pytest app/tests/test_retry_mechanism.py -v

# RAG service tests
python -m pytest app/tests/test_rag_service.py -v

# API endpoint tests
python -m pytest app/tests/test_api_endpoints.py -v

# Session service tests
python -m pytest app/tests/test_session_service.py -v
```

## Test Environment

All tests use **comprehensive mocking** to avoid dependencies on external services:

- **MongoDB**: Mocked async/sync database connections and operations
- **Vertex AI**: Mocked Google Cloud AI services (TextEmbeddingModel, aiplatform)
- **Gemini AI**: Mocked genai.Client for conversation guide and RAG services
- **Authentication**: Mocked Google Cloud credentials and authentication

### Environment Variables (Automatically Set)
The tests automatically set these test environment variables:
```python
MONGODB_URI = "mongodb://test:27017"
GOOGLE_CLOUD_PROJECT = "test-project"
MONGODB_DATABASE = "test_db"
MONGODB_COLLECTION = "test_collection"
VERTEX_AI_LOCATION = "test-location"
```

**Important**: You do NOT need to provide real MongoDB URIs or Vertex AI credentials for testing!

## Test Coverage

### ConversationGuideService Tests
- ✅ Quality analysis for factual statements, emotions, and detailed expressions
- ✅ Cumulative input processing across multiple conversation rounds
- ✅ Guidance response generation with appropriate prompts
- ✅ Ready-for-search determination based on quality scores
- ✅ Mocked Gemini AI responses for different input scenarios

### Retry Mechanism Tests
- ✅ Retry on ResourceExhausted (429) errors with exponential backoff
- ✅ Retry on ServiceUnavailable (503) errors
- ✅ Retry on DeadlineExceeded (504) errors
- ✅ No retry on ValueError (non-retryable errors)
- ✅ Max retries exceeded behavior (3 attempts)
- ✅ LRU caching behavior during retries
- ✅ Empty input handling with early return

### RAG Service Tests
- ✅ Emotion statistics calculation with valence-arousal filtering
- ✅ Primary emotion extraction from complex emotion labels
- ✅ Search results processing with enriched emotion analysis
- ✅ Error handling for Gemini AI API failures
- ✅ Service initialization and configuration

### API Endpoint Tests
- ✅ Health check endpoint with database connectivity status
- ✅ Search endpoint with session management and conversation guide integration
- ✅ Search execution endpoint with RAG analysis and result processing
- ✅ Error handling for missing sessions and invalid input
- ✅ Input validation and HTTP status code verification

### Session Service Tests
- ✅ Session creation with UUID generation and timestamp tracking
- ✅ Session retrieval by ID with proper error handling
- ✅ Session updates with modified timestamp tracking
- ✅ Database unavailable scenarios with graceful degradation
- ✅ Service initialization with null database handling

## Key Features

1. **No External Dependencies**: All tests run with comprehensive mocking
2. **Complete Service Coverage**: Tests cover all major backend components
3. **Error Scenario Testing**: Tests include both success and failure paths
4. **Async Operation Support**: Proper handling of async/await patterns
5. **Detailed Test Output**: Clear success/failure indicators and error messages
6. **Performance Considerations**: Tests run quickly without network calls

## Test Results Summary

When running `run_all_tests.py`, you'll see a summary like:
```
============================================================
TEST SUMMARY
============================================================
test_retry_mechanism      ✅ PASSED
test_rag_service          ✅ PASSED
test_api_endpoints        ✅ PASSED
test_session_service      ✅ PASSED
async_tests               ✅ PASSED

Total: 5/5 tests passed
🎉 All tests passed!
```

## Adding New Tests

When adding new tests:

1. **Use Comprehensive Mocking**: Mock all external services (MongoDB, Vertex AI, Gemini AI)
2. **Set Environment Variables**: Use test-specific environment variables
3. **Test Both Scenarios**: Include both success and error cases
4. **Clear Descriptions**: Use descriptive test names and docstrings
5. **Update Test Runners**: Add new tests to `run_all_tests.py`
6. **Update Documentation**: Update this README with new test categories

## Troubleshooting

If you see MongoDB or Vertex AI connection errors during testing:
1. Check that all necessary services are mocked in your test
2. Ensure environment variables are set before importing services
3. Verify that mocks are applied before the service initialization
4. Remember: tests should NEVER connect to real external services!

## Prerequisites

Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

## Notes

- All tests use mocking to avoid external dependencies
- Async tests require proper async/await handling with pytest-asyncio
- Database tests use mock collections and clients
- API tests use FastAPI's TestClient with comprehensive service mocking
- Tests are designed to run quickly and reliably in any environment 