def chunk_text(text,max_size=100) :
    chunk = [] 
    words=text.split()
    for i in range(0,len(words),max_size) :
        chunk.append(" ".join(words[i:i+max_size]))
    return chunk