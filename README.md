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

Deeper into Chunking Strategies
Let’s delve further into this concept by exploring the critical considerations when deciding on a chunking strategy.

#### 1. Nature of Content
The type of content you’re dealing with can significantly influence your chunking strategy. For example, are you processing long-form content like articles and books? Or are you working with bite-sized information like tweets and instant messages?

Think of this as the difference between solving a 1000-piece puzzle and a 100-piece puzzle. The former requires more intricate chunking, while the latter might not need any at all.

Your answer will help guide your choice of model, and in turn, determine your chunking strategy.

#### 2. Choice of Embedding Model
The type of embedding model you use also plays a role in your chunking strategy. Different models perform better with different chunk sizes.

To stick with our puzzle analogy, it’s like using different strategies for a landscape puzzle versus a detailed portrait puzzle. Sentence-transformer models, for example, might work better on individual sentences, similar to how focusing on individual colors might help solve the landscape puzzle. On the other hand, a model like text-embedding-ada-002 might perform better on chunks containing 256 or 512 tokens, akin to focusing on facial features in a portrait puzzle.

#### 3. User Query Expectations
How complex do you anticipate your user queries to be? Will users typically input short and specific queries, or will they be longer and more complex?

Let’s relate this back to searching for a puzzle piece. If you’re searching for an edge piece (a simple, specific query), you would use a different method than if you were searching for a piece with a specific pattern that could be anywhere in the puzzle (a complex query). Your chunking strategy should accommodate these different search methods to increase the chances of a successful match.

#### 4. Utilization of Retrieved Results
Finally, consider how the results retrieved from your chunking strategy will be used. Will they contribute to a semantic search, answer questions, aid in summarizing content, or serve another purpose?

Say you’ve completed sections of your puzzle and need to fit them together. The way you assemble them will depend on the final picture you want to achieve. Similarly, if your results need to be input into another LLM with a token limit, you’ll need to adjust your chunk sizes accordingly to ensure they can fit into the request.

By asking these questions, you can fine-tune a chunking strategy that balances performance and accuracy, ultimately improving the relevance of your query results.




#### Chunk Size:

The size of the chunks can have a significant impact on the quality of the RAG system. While large sized chunks provide better context, they also carry a lot of noise. Smaller chunks, on the other hand, have precise information but they might miss important information. For instance, consider a legal document that’s 10,000 words long. If we chunk it into 1,000-word segments, each chunk might contain multiple legal clauses, making it hard to retrieve specific information. Conversely, chunking it into 200-word segments allows for more precise retrieval of individual clauses, but may lose the context provided by surrounding clauses. Experimenting with chunk sizes can help find the optimal balance for accurate retrieval. The processing time also depends on the size of the chunk. Chunk size, therefore, has a significant impact on retrieval accuracy, processing speed and storage efficiency. The ideal chunk size varies with the use case and depends on balancing factors like document types and structure, complexity of user query and the desired response time. There is no one size fits all approach towards optimizing chunk sizes. Experimentation and evaluation of different chunk sizes on metrics like faithfulness, relevance, response time can help in identifying the optimal chunk size for the RAG system. Chunk size optimisation may require periodic reassessment as data, or requirements change.

#### Nature of the Content:

The type of data that you’re dealing with can be a guide for the chunking strategy. If your application uses a data in a specific format like code or HTML, a specialized chunking method is recommended. Not only that, whether you’re working with long documents like whitepapers and reports or short form content like social media posts, tweets, etc. can also guide the chunk size and overlap limits. If you’re using a diverse set of information sources, then you might have you use different methods for different sources.

#### Expected Length and Complexity of User Query:
The nature of the query that your RAG enabled system is likely to receive is also a determinant of the chunking strategy. If your system expects a short and straightforward query, then the size of your chunks should be different when compared to a long and complex query. Matching long queries to short chunks may prove inefficient in certain cases. Similarly, short queries matching with large chunks may yield partially irrelevant results.


#### Enhancements in the context awareness of chunks

Adaptive chunking that adjusts chunks dynamically based on the use case.

Evolution of agentic chunking into a more sophisticated task-oriented process.

Multimodal chunking is designed to segment not only text but other unstructured form of data like images, audio and video.

Deeper integration with knowledge graphs to maintain connections to broader concepts.

Real-time and responsive chunking for edge devices and low-latency systems.

Chunking is a crucial step towards creating production grade RAG systems. While fixed width chunking methods are good for prototyping and developing simpler systems, more advanced strategies are required from production grade applications. As chunking evolves with the evolution of generative AI, it is important to develop points of view on chunking strategies and experiment with the ones available for different use cases.




.



# Here is a detailed breakdown of chunking strategies, including their advantages, use cases, and code examples.





'


---

### Fixed Size Chunking

A very common approach is to pre-determine the size of the chunk and the amount of overlap between the chunks. There are several chunking methods that follow a fixed size chunking approach.

**Character-Based Chunking:** Chunks are created based on a fixed number of characters

**Token-Based Chunking:** Chunks are created based on a fixed number of tokens.

**Sentence-Based Chunking:** Chunks are defined by a fixed number of sentences

**Paragraph-Based Chunking:** Chunks are created by dividing the text into a fixed number of paragraphs.

While character and token based chunking methods are useful to maintain consistency of chunk sizes, they may lead to loss of meaning by abruptly splitting words or sentences. On the other hand, while sentence and paragraph based chunking methods are helpful in preserving the context, they introduce a variability in chunk sizes.

An example of fixed size chunking using LangChain is shown below
```python
#import libraries
from langchain_text_splitters import CharacterTextSplitter
#Set the CharacterTextSplitter parameters
text_splitter = CharacterTextSplitter(
    separator="\n",    #The character that should be used to split
    chunk_size=1000,   #Number of characters in each chunk
    chunk_overlap=200, #Number of overlapping characters between chunks
)

#Create Chunks
chunks=text_splitter.create_documents([data_transformed[0].page_content])

#Show the number of chunks created
print(f”The number of chunks created : {len(chunks}”)
```

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

### 10.Fixed-Size Chunking:The Bread-and-Butter Technique
Fixed-size chunking is akin to slicing bread. You decide how thick you want each slice (or chunk) to be and start cutting, occasionally allowing for overlaps to ensure the entire loaf is sliced.

Fixed-size chunking is favored for its simplicity and computational efficiency, as it doesn’t rely on complex NLP libraries. It’s your go-to, bread-and-butter technique for most common use cases.

Here’s an illustrative code example:
```python

text = "..." # your text
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 256,
    chunk_overlap  = 20
)
docs = text_splitter.create_documents([text])
```
In this example, we first define the content to be chunked (text). Then, we use the CharacterTextSplitter from LangChain to set our separator, chunk size, and overlap. The result (docs) will be a sequence of chunks, each containing roughly 256 tokens with an overlap of 20 tokens.





### 11.Content-Aware Chunking: The Tailor-Made Approach

Think of content-aware chunking as custom-tailoring a suit instead of buying one off the rack. It molds the chunks according to the content’s nature, resulting in a more refined output. This approach can be broken down into several sub-methods, like sentence splitting.

Naive splitting is the simplest approach, akin to tearing a loaf of bread into chunks instead of neatly slicing it. It just divides sentences at every period or new line, but might not account for edge cases:
```python
text = "..." # your text
docs = text.split(".")
The NLTK library, like a more precise knife, offers a more nuanced approach, carving out sentence chunks while preserving meaningful context:

text = "..." # your text
from langchain.text_splitter import NLTKTextSplitter

text_splitter = NLTKTextSplitter()
docs = text_splitter.split_text(text)
The spaCy library offers a sophisticated approach, similar to a scalpel, excelling at carving precise sentences while maintaining context:

Recursive Chunking: The Russian Doll Technique
Recursive chunking is like a Russian nesting doll. If the text isn't divided into suitable chunks at the first attempt, the method recursively repeats the process until it achieves the desired chunk size or structure:

text = "..." # your text
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 256,
    chunk_overlap  = 20
)

docs = text_splitter.create_documents([text])
```

### 12.Context-Enriched Chunking

This method adds the summary of the larger document to each chunk to enrich the context of the smaller chunk. This makes more context available to the LLM without adding too much noise. It also improves the retrieval accuracy and maintains semantic coherence across chunks. This is particularly useful in scenarios where a more holistic view of the information is crucial. While this approach enhances the understanding of the broader context, it adds a level of complexity and comes at the cost of higher computational requirements, increased storage needs and possible latency in retrieval. Below is an example of how context enrichment can be done using GPT-4o-mini, OpenAI embeddings and FAISS.
```python
#Loading text from Wikipedia page
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
url=https://en.wikipedia.org/wiki/2023_Cricket_World_Cup
loader = AsyncHtmlLoader (url)
data = loader.load()
html2text = Html2TextTransformer()
document_text=data_transformed[0].page_content

# Generating summary of the text using GPT-4o-mini model
summary_prompt = f"Summarize the given document in a single /
paragraph\ndocument: {document_text}" 
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages= [
    {"role": "user", "content": summary_prompt}
      ]
)

summary=response.choices[0].message.content

# Creating Chunks using Recursive Character Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1000,
chunk_overlap=200)
chunks=text_splitter.split_text(data_transformed[0].page_content)

# Enriching Chunks with Summary Data
context_enriched_chunks = [answer + "\n" + chunk for chunk in chunks]

# Creating embeddings and storing in FAISS index
embedding = OpenAIEmbeddings(openai_api_key=api_key) #E
vector_store = FAISS.from_texts(context_enriched_chunks, embedding)
```




### 13.Specialized Chunking: The Master Craftsman Approach

This method is for specific, structured content like Markdown and LaTeX, which require dedicated techniques to preserve their original structure during chunking. It's akin to a master craftsman using specialized tools to create a work of art:

Markdown chunking:
```python
from langchain.text_splitter import MarkdownTextSplitter
markdown_text = "..."

markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
docs = markdown_splitter.create_documents([markdown_text])
LaTeX chunking:

from langchain.text_splitter import LatexTextSplitter
latex_text = "..."

latex_splitter = LatexTextSplitter(chunk_size=100, chunk_overlap=0)
docs = latex_splitter.create_documents([latex_text])
```
Determining Your Optimal Chunk Size: The Goldilocks Principle
Choosing the ideal chunk size is an art form. It's akin to finding the porridge that's "just right" in the Goldilocks tale. You'll have to experiment with various sizes, evaluate performance, and refine your approach iteratively until you achieve the best balance between context preservation and accuracy.




### 14.Structure-Based Chunking:

The aim of chunking is to keep meaningful data together. If we are dealing with data in form of HTML, Markdown, JSON or even computer code, it makes more sense to split the data based on the structure rather than a fixed size. Another approach for chunking is to take into consideration the format of the extracted and loaded data. A markdown file, for example is organised by headers, a code written in a programming language like python or java is organized by classes and functions and HTML, likewise, is organised in headers and sections. For such formats a specialised chunking approach can be employed.

An example of chunking based on HTML headers is as follows —
```python
# Import the HTMLHeaderTextSplitter library
from langchain_text_splitters import HTMLHeaderTextSplitter

# Set url as the Wikipedia page link
url="https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

# Specify the header tags on which splits should be made
headers_to_split_on=[
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4")
]

# Create the HTMLHeaderTextSplitter function
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Create splits in text obtained from the url
html_header_splits = html_splitter.split_text_from_url(url)
```
Advanced Chunking Techniques
Both fixed size and structured chunking methods fail to take semantic context into account. They don’t consider the actual meaning on the content. To address these limitations, several advanced techniques have been introduced.




### 15.Agentic Chunking:

In agentic chunking, chunks from the text are created based on a goal or a task. Consider an e-commerce platform wanting to analyse customer reviews. The best way for the reviews to be chunked is if the reviews pertaining to a particular topic are put in the same chunk. Similarly, the critical reviews and positive reviews may be put in different chunks. To achieve this kind of chunking, we will need to do sentiment analysis, entity extraction and some kind of clustering. This can be achieved by a multi-agent system. Agentic chunking is still an active area of research and improvement.



### Conclusion

Each chunking strategy has its strengths and weaknesses, and the choice of strategy depends on the task at hand. If you need more control over chunk size, character-based and token-based strategies are effective. However, if you want to preserve meaning and semantic flow, sentence-based or semantic chunking is preferable. For more advanced use cases like question answering, dynamic and hybrid chunking strategies provide flexibility.
