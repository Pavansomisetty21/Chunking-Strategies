# Chunking-Strategies

Large language models (LLMs) do not have all the information you need, especially for specific topics like your company’s data. RAG helps LLMs provide better answers on topics they are not familiar with by using the user’s organization’s specific data. The effectiveness of RAG systems hinges on one crucial factor: Data Parsing. We discussed the potential pitfalls of using improperly parsed data, including irrelevant retrieval, missing information, and reduced accuracy. We also discussed strategies for structuring data for retrieval because the success of a RAG system fundamentally depends on the quality of its data foundation. 

The goal of chunking is, as its name says, to chunk the information into multiple smaller pieces in order to store it in a more efficient and meaningful way. This allows the retrieval to capture pieces of information that are more related to the question at hand, and the generation to be more precise, but also less costly, as only a part of a document will be included in the LLM prompt, instead of the whole document.

### What is Chunking in RAG?
The role of chunking in RAG and the underlying idea is somewhat similar to what it is in real life. Once you’ve extracted and parsed text from the source, instead of committing it all to memory as a single element, you break it down into smaller chunks.

### Importance of Chunking in RAG

1. **Context Window Limitations**: Allows data to fit within the limited input context size of language models, ensuring effective processing.
2. **Cost Efficiency**: Reduces the number of tokens sent to the model, lowering costs associated with querying large documents.
3. **Relevance**: Helps retrieve and process only the most pertinent information, improving accuracy and relevance.
4. **Improved Accuracy**: Enables more focused and granular retrieval, minimizing irrelevant content.
5. **Faster Response Times**: Allows for parallel processing and retrieval, enhancing system responsiveness.
6. **Efficient Indexing**: Makes indexing and storage more scalable and manageable for large datasets.
7. **Better User Experience**: Ensures faster and more relevant answers, improving interaction.
8. **Context Preservation**: Maintains relevance across queries by providing focused and relevant chunks.
9. **Facilitates Dynamic Updates**: Allows easy updating and retrieval of specific chunks, making the system more adaptable.
10. **Reduces Model Overload**: Simplifies decision-making by avoiding large, overwhelming inputs, leading to better model comprehension.

Chunking is essential for optimizing performance, relevance, and efficiency in RAG systems.

### Chunk Size:

**Too small:** Can lead to a loss of context and hinder the LLM’s ability to understand the relationships between ideas.

**Too large:** Exceeds the LLM’s context window, reducing efficiency and potentially degrading performance.

### Chunk Overlap:

**Purpose:** Overlap ensures that information is covered from multiple perspectives, improving the chances of capturing relevant context.

**Optimal overlap:** The ideal overlap percentage depends on factors like document length, LLM capabilities, and desired level of precision.

**Example:** For a 200-page technical manual, one can choose a chunk size of 500 words with a 20% overlap. This means each chunk is 500 words, and the subsequent chunk starts 100 words into the previous chunk.

## Exploring Different Chunk Sizes
Throughout this article, we’ll break down various chunking methods, their trade-offs, and how to choose the most suitable one for your application.

The Difference Between Short and Long Embeddings
When embedding content, we can expect different outcomes based on whether the content is short, like a sentence, or long, like a paragraph or an entire document.

### Short Content:
If we look at a sentence-level embedding, it’s like focusing on a single puzzle piece. It concentrates on the specific meaning of that piece (sentence) but might overlook the bigger picture (the entire puzzle or document).

### Long Content:
On the other hand, embedding a whole paragraph or document is like trying to solve a larger section of the puzzle. This process takes into account the overall image (context) and how the individual pieces (sentences or phrases) relate to each other. While this provides a broader perspective, it might also introduce ‘noise’ or dilute the importance of individual pieces (sentences).

### Considerations for Queries
The length of your query also influences how the embeddings relate to each other. A short query is like looking for a specific puzzle piece, better suited for matching against sentence-level embeddings. In contrast, a longer query seeks a broader context or theme, making it more aligned with paragraph or document-level embeddings.


## Chunking Strategies / Selecting a Chunk Size

Here is a detailed breakdown of chunking strategies used in LangChain, including their advantages, use cases, and code examples.

---

### 1. **Character-Based Chunking (`CharacterTextSplitter`)**

#### **Description:**
- **Character-based chunking** involves splitting text based on a specified number of characters.
- It is simple and effective for scenarios where you need a consistent chunk size, especially when working with models that expect uniform input lengths.

#### **Advantages:**
- Easy to implement.
- Useful when exact chunk size is important.
  
#### **Disadvantages:**
- May cut sentences, disrupting the semantic flow.

#### **Example Code:**

```python
from langchain.text_splitter import CharacterTextSplitter

text = "This is a long text that we want to chunk based on character length for better processing."

# Initialize a character-based text splitter
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)

# Split the text into chunks
chunks = text_splitter.split_text(text)
print(chunks)
```

#### **Explanation:**
- `chunk_size`: Number of characters in each chunk.
- `chunk_overlap`: Number of characters that overlap between consecutive chunks.

---

### 2. **Token-Based Chunking (`RecursiveCharacterTextSplitter`)**

#### **Description:**
- Token-based chunking splits the text based on tokens, such as words or subwords (common with transformer models).
- This method ensures that the text chunks respect the model’s token limitations.

#### **Advantages:**
- Ensures that text chunks fit within the token limit of models like GPT-3 or BERT.
- Prevents cutting words or sentences awkwardly.

#### **Disadvantages:**
- Can be slower due to tokenization overhead.

#### **Example Code:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "This text will be split using token-based chunking to ensure each chunk stays within token limits."

# Initialize a token-based splitter with recursive chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

chunks = text_splitter.split_text(text)
print(chunks)
```

#### **Explanation:**
- The `RecursiveCharacterTextSplitter` attempts to split at sentence boundaries, words, and characters, ensuring that chunks stay within the specified token limit.

---

### 3. **Sentence-Based Chunking (`SentenceTextSplitter`)**

#### **Description:**
- Sentence-based chunking splits text at sentence boundaries, ensuring that each chunk contains full sentences. This makes it useful when maintaining the semantic integrity of text is important.

#### **Advantages:**
- Preserves the meaning of text by keeping sentences intact.
  
#### **Disadvantages:**
- Less control over the chunk size, as sentences may vary in length.

#### **Example Code:**

```python
from langchain.text_splitter import SentenceTextSplitter

text = "This is sentence one. Here is sentence two. Now we have a third sentence."

# Initialize sentence-based text splitter
text_splitter = SentenceTextSplitter(chunk_size=2)

chunks = text_splitter.split_text(text)
print(chunks)
```

#### **Explanation:**
- `chunk_size`: Maximum number of sentences per chunk.

---

### 4. **Document-Based Chunking (`DocumentTextSplitter`)**

#### **Description:**
- This strategy chunks documents based on sections or metadata such as headings, paragraphs, or logical divisions within structured documents.

#### **Advantages:**
- Effective for structured documents (e.g., articles, reports, legal documents).
- Allows splitting based on logical document parts (e.g., headings or metadata).

#### **Disadvantages:**
- May not be ideal for unstructured text or free-flowing prose.

#### **Example Code:**

```python
from langchain.text_splitter import DocumentTextSplitter

# Example document structure
documents = [
    {"text": "Title: Section 1\nThis is the content of section one.", "metadata": {"section": "Introduction"}},
    {"text": "Title: Section 2\nThis is the content of section two.", "metadata": {"section": "Body"}}
]

# Initialize a document text splitter
text_splitter = DocumentTextSplitter(chunk_size=50, chunk_overlap=10)

chunks = text_splitter.split_documents(documents)
for chunk in chunks:
    print(chunk['text'])
```

#### **Explanation:**
- `chunk_size`: Defines the size of chunks (usually based on characters or tokens).
- `chunk_overlap`: Allows overlapping between sections of a document.

---

### 5. **Paragraph-Based Chunking**

#### **Description:**
- This method splits text based on paragraph boundaries, making it useful for structured content like blogs, essays, or legal documents where paragraphs are distinct units of thought.

#### **Advantages:**
- Useful for preserving the logical flow of paragraphs.
  
#### **Disadvantages:**
- Chunks may vary greatly in size if paragraphs differ significantly in length.

#### **Example Code:**

```python
from langchain.text_splitter import ParagraphTextSplitter

text = """
This is paragraph one.
It contains some information.

This is paragraph two.
It contains more information.
"""

# Initialize paragraph-based splitter
text_splitter = ParagraphTextSplitter(chunk_size=1)

chunks = text_splitter.split_text(text)
print(chunks)
```

#### **Explanation:**
- This splits text into distinct paragraphs, preserving meaning and logical structure.

---

### 6. **Semantic Chunking (Meaning-Based)**

#### **Description:**
- Semantic chunking splits text based on topic shifts, context, or meaning, usually done using embeddings or models that understand the text’s content.

#### **Advantages:**
- Splits text into semantically coherent chunks.
- Enhances the retrieval of relevant information for question-answering tasks.

#### **Disadvantages:**
- Requires more complex tools such as embeddings or pre-trained models.

#### **Example Code (Custom):**

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

text = "This is a long document split based on semantic meaning and context. Here is a new section with distinct meaning."

# Initialize text loader and embeddings for semantic chunking
loader = TextLoader(text)
docs = loader.load()
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store to retrieve semantically similar chunks
vector_store = FAISS.from_documents(docs, embeddings)

# Chunk the document based on semantic similarity
query = "Context related to specific meaning"
similar_docs = vector_store.similarity_search(query, k=3)

for doc in similar_docs:
    print(doc.page_content)
```

#### **Explanation:**
- Semantic chunking identifies semantically coherent chunks of text. This example uses FAISS and embeddings to retrieve semantically related portions of text based on a query.

---

### 7. **Dynamic Chunking (Query-Based Chunking)**

#### **Description:**
- This strategy dynamically selects and chunks text based on a user query, ensuring that only relevant chunks are retrieved.
  
#### **Advantages:**
- Highly relevant chunks are returned, making it ideal for question-answering and retrieval-augmented generation (RAG).
  
#### **Disadvantages:**
- Requires embedding models and a vector store for implementation.

#### **Example Code (Using FAISS):**

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

# Load documents and initialize embedding
loader = TextLoader("path_to_document.txt")
docs = loader.load()
embeddings = OpenAIEmbeddings()

# Create FAISS vector store from documents
vector_store = FAISS.from_documents(docs, embeddings)

# Perform query-based chunking
query = "What is the importance of chunking in RAG?"
relevant_chunks = vector_store.similarity_search(query, k=3)

for chunk in relevant_chunks:
    print(chunk.page_content)
```

#### **Explanation:**
- This method dynamically chunks the text based on the relevance of the text to a specific query, allowing you to retrieve only the most pertinent information.

---

### 8. **Sliding Window Chunking**

#### **Description:**
- Sliding window chunking creates overlapping chunks of text. This helps retain context between chunks, ensuring that important information from the end of one chunk is carried over to the next.

#### **Advantages:**
- Retains context between chunks, improving the model's ability to understand the whole document.

#### **Disadvantages:**
- Increased token usage due to overlap between chunks.

#### **Example Code:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "This is a long document where we apply sliding window chunking to ensure the model retains context."

# Initialize a splitter with chunk overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

chunks = text_splitter.split_text(text)
print(chunks)
```

#### **Explanation:**
- `chunk_overlap`: Specifies how many characters or tokens overlap between consecutive chunks.

---

### 9. **Hybrid Chunking (Multi-Strategy Approach)**

#### **Description:**
- Hybrid chunking combines multiple strategies (e.g., sentence + token-based, paragraph + semantic) to optimize chunking based on document type or use case.

#### **Advantages:**
- Flexible and customizable.
- Can adapt to various text types and tasks.

#### **Disadvantages:**
- More complex to implement.

#### **Example Code (Custom Hybrid Strategy):**

```python
from langchain.text_splitter import SentenceTextSplitter, RecursiveCharacterTextSplitter

text = "This is a long document. We'll use both sentence-based and token-based chunking for efficiency."

# Sentence-based splitter
sentence_splitter = SentenceTextSplitter(chunk_size=2)
sentence_chunks = sentence_splitter.split_text(text)

# Token

-based chunking on each sentence chunk
final_chunks = []
token_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

for chunk in sentence_chunks:
    final_chunks.extend(token_splitter.split_text(chunk))

print(final_chunks)
```

#### **Explanation:**
- This hybrid approach first chunks the text into sentences and then applies token-based chunking within each sentence chunk for flexibility and control.

---

### Conclusion

Each chunking strategy has its strengths and weaknesses, and the choice of strategy depends on the task at hand. If you need more control over chunk size, character-based and token-based strategies are effective. However, if you want to preserve meaning and semantic flow, sentence-based or semantic chunking is preferable. For more advanced use cases like question answering, dynamic and hybrid chunking strategies provide flexibility.
