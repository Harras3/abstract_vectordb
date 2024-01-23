from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SentenceWindowNodeParser
#from helper_functions import load_pdf
import os
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index import load_index_from_storage



class llama_index_db:
    query_engine=False
    sentence_window_engine=''
    sentence_index=''
    def __init__ (self): 
        return 
    
    def insert_in_db(self,
    documents,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index",
    ):
 
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        sentence_context = ServiceContext.from_defaults(
            #llm=llm,
            embed_model=embed_model,
            node_parser=node_parser,
        )
        
        if not os.path.exists(save_dir):
            self.sentence_index = VectorStoreIndex.from_documents(
                documents, service_context=sentence_context
            )
            self.sentence_index.storage_context.persist(persist_dir=save_dir)
        else:
            self.sentence_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=save_dir),
                service_context=sentence_context,
            )

        return self.sentence_index

    
 
    
    def query(self, query,similarity_top_k=6, rerank_top_n=2
    ):
        
        if self.query_engine==False:
            postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
            rerank = SentenceTransformerRerank(
                top_n=rerank_top_n, model="BAAI/bge-reranker-base"
            )

            self.sentence_window_engine = self.sentence_index.as_query_engine(
                similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
            )
            self.query_engine=True
        ans = self.sentence_window_engine.query(query)
        return ans




