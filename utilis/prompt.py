def build_prompt(context_chunks,query):
    context ="\n\n".join(context_chunks)
    return f""" Use the following context to answer the question

context:
{context}

Question:
{query}

Answer:"""