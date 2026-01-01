# Learning the basics of chunking
Chunking is the process of dividing up data/information before embedding it using a model such as a sentence model.
Once we have chunked data, it is embedded into vectors that collapse down to an individual vector per chunk.
We then compare this vector with our query for similarity, using the result to select answers to questions.

We have three scripts to help implement some simple chunking:

**chunker.py**
contains a `Chunk` class for representing a given chunk of data.
provides functions for different chunking strategies such as fixed size, structure-aware.
fixed size simply chunks by a given number of tokens (overlap can be used)
structure-aware chunking works best on documents such as html, markdown, yaml, to break into chunks based on the inherent structure of the file - which should already be quite informative.

**embedder.py**
has functions for embedding queries or chunks into vectors, comparing a query with chunks for a best match, and implementing a basic similarity.
makes use of a simple sentence model to handle the encoding of chunks.  similarity can also be used, however we have implemented it ourselves.

**tester.py**
read a test document (in this case a markdown file full of information about markdown documents)
create a query
use the `best_match()` function from the embedder to select the most ideal chunk for the query.
