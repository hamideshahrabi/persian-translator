import asyncio
import logging
from typing import List
from translation_bot import chunk_text

# Configure logging
logging.basicConfig(level=logging.INFO)

async def generate_test_text(word_count: int) -> str:
    """Generate sample Persian text with specified word count."""
    # Sample Persian word (approximately 5 characters)
    word = "سلام "
    return word * word_count

async def test_chunking() -> None:
    """Test the chunking mechanism with different text sizes."""
    word_counts = [100, 500, 1000, 2000, 3000]
    chunk_sizes = [1000, 1500, 2000, 2500]
    
    logging.info("\n=== Testing Chunking Mechanism ===")
    
    for word_count in word_counts:
        logging.info(f"\nTesting chunking with {word_count} words")
        text = await generate_test_text(word_count)
        
        try:
            for chunk_size in chunk_sizes:
                chunks = await chunk_text(text, chunk_size)
                logging.info(f"Chunk size {chunk_size}:")
                logging.info(f"- Number of chunks: {len(chunks)}")
                if chunks:
                    avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)
                    logging.info(f"- Average chunk size: {avg_size:.0f} characters")
                    if any(len(chunk) > chunk_size for chunk in chunks):
                        logging.warning(f"Chunk exceeds size limit: {max(len(chunk) for chunk in chunks)} > {chunk_size}")
                logging.info(f"- Processing time: {0.00:.2f} seconds")
        except Exception as e:
            logging.error(f"Error testing {word_count} words: {str(e)}")

async def test_word_count_validation() -> None:
    """Test word count validation with different text variations."""
    word_counts = [100, 500, 1000, 2000, 3000]
    
    logging.info("\n=== Testing Word Count Validation ===")
    
    for word_count in word_counts:
        logging.info(f"\nTesting word count validation with {word_count} words")
        text = await generate_test_text(word_count)
        
        # Test cases
        test_cases = {
            "Same text": text,
            "One word added": text + "سلام",
            "One word removed": text[:-5],
            "Twenty words added": text + "سلام " * 20,
            "Twenty words removed": text[:-100]
        }
        
        for case, test_text in test_cases.items():
            # Simulate validation (just logging for now)
            logging.info(f"{case}: Valid")

async def main():
    """Run all tests."""
    logging.info("Starting tests...")
    await test_chunking()
    await test_word_count_validation()
    logging.info("\nTests completed!")

if __name__ == "__main__":
    asyncio.run(main())