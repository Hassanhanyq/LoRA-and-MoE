from chonkie import NeuralChunker
chunker = NeuralChunker(
    model="mirth/chonky_modernbert_base_1",  
    device_map="cpu",                        
    min_characters_per_chunk=10,             
)
def chunk_documents(texts):
    """
    chunk documents and stuff
    """
    return chunker.chunk_batch(texts)