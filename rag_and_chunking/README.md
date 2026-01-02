# Learning the basics of chunking
Chunking is the process of dividing up data/information before embedding it
using a model such as a sentence model. Once we have chunked data, it is
embedded into vectors that collapse down to an individual vector per chunk. We
then compare this vector with our query for similarity, using the result to
select answers to questions.

We have three scripts to help implement some simple chunking:

**chunker.py** contains a `Chunk` class for representing a given chunk of data.
provides functions for different chunking strategies such as fixed size,
structure-aware. fixed size simply chunks by a given number of tokens (overlap
can be used) structure-aware chunking works best on documents such as html,
markdown, yaml, to break into chunks based on the inherent structure of the file
- which should already be quite informative.

**embedder.py** has functions for embedding queries or chunks into vectors,
comparing a query with chunks for a best match, and implementing a basic
similarity. makes use of a simple sentence model to handle the encoding of
chunks.  similarity can also be used, however we have implemented it ourselves.

**tester.py** read a test document (in this case a markdown file full of
information about markdown documents) create a query use the `best_match()`
function from the embedder to select the most ideal chunk for the query.

# Flaws
**semantic collision** one of the most obvious flaws with chunking and then
comparing with a query is `semantic collision`. this is where our sentence model
may present a "best match" because it has found similar tokens, but the meaning
in the chunk is not relevant.

we can see an example of this in when querying against our test doc.  if we
query with "please block that quote" - as though looking to have something
removed from a document, our model's best matches talking about block quotes in
markdown formatting. 

**chunking strategy weaknesses** using something like structure-aware chunking
seems logical for our markdown-formatted test doc, as the structure of the
document implies something about the meaning of the document - however this can
(and in our case does) result in a large variance in the amount of
data/information in chunks.  sometimes a chunk is one sentence, sometimes it is
multiple paragraphs.  this heavily limits the accuracy of the results - for
example querying "what restrictions are at block level?" should get a near
perfect match in our 'Inline HTML' section, but because the chunk is so big it
seems likely that the model is failing to pick up on that - and is instead
returning much less appropriate responses.
