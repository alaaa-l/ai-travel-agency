from embeddings import embed_texts
from similarity import retrieve_relevant_chunks
from prompt import prepare_prompt
from call_llm import generate_answer
import os

def resolve_iata_code(location_name: str, rag_collection) -> dict:
    """
    Resolves a country or city name to IATA city and airport codes
    using RAG (authoritative data only).
    """

    # 1. Embed query
    query_vector = embed_texts([location_name])

    # 2. Retrieve relevant chunks
    result = retrieve_relevant_chunks(
        query_vector,
        rag_collection,
        top_k=5
    )

    # 3. Strict prompt 
    system_prompt = """
    You are a location code resolver.
    Use ONLY the provided context.
    
    Extract:
    - City name
    - IATA city code (if exists)
    - IATA airport codes

    If multiple airports exist, list all.
    If no code exists, say "NOT FOUND".

    Return STRICT JSON:
    {
      "city": "",
      "iata_city": "",
      "airports": []
    }
    """

    context = "\n".join(result["documents"][0])

    final_prompt = f"""
    Context:
    {context}

    Location query:
    {location_name}
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")

    response = generate_answer(
        system_prompt + "\n" + final_prompt,
        api_key
    )

    return response
