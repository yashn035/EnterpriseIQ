import time
import hashlib
from cachetools import TTLCache
from src.retrieval.local_retriever import LocalRetriever

class DummyCounter:
    def inc(self, *args, **kwargs):
        pass
    def labels(self, *args, **kwargs):
        return self

RAG_ERRORS_TOTAL = DummyCounter()

class RAGChain:
    def __init__(self):
        self.retriever = LocalRetriever()
        self.cache = TTLCache(maxsize=100, ttl=3600)
        self.circuit_breaker = {"failures": 0, "last_failure": 0}
    
    def _mock_llm(self, context, question):
        if not context:
            return "I don't have this information."
        
        q_lower = question.lower()
        selected = context[0]
        for c in context:
            txt_lower = c["text"].lower()
            if "leave" in q_lower and "leave" in txt_lower:
                selected = c
                break
            elif "remote" in q_lower and "remote" in txt_lower:
                selected = c
                break

        sources = list(set([c["source"] for c in context]))
        return f"Based on the policy documents, the answer is: {selected['text']} [Source: {selected['source']}]"
    
    def ask(self, question, request_id="N/A"):
        start_time = time.time()
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if cache_key in self.cache:
            return {**self.cache[cache_key], "latency_ms": 1.0, "cached": True}
        
        context = self.retriever.retrieve(question, top_k=3)
        answer = self._mock_llm(context, question)
        sources = list(set([c["source"] for c in context])) if context else []
        
        result = {"answer": answer, "sources": sources, "latency_ms": round((time.time() - start_time) * 1000, 2)}
        self.cache[cache_key] = result
        return result

    def warmup(self):
        pass
