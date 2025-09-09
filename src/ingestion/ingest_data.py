import os
import json
import chromadb
from typing import List, Dict, Any

class DataIngester:
    def __init__(self, data_dir: str = "data", db_path: str = "./chroma_db"):
        """Initialize the data ingester"""
        self.data_dir = data_dir
        self.db_path = db_path
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="byd_seal_docs",
            metadata={"description": "RAG data collection for BYD Seal"}
        )
        

    
    def ingest_all_data(self):
        """Ingest all data files from the data directory"""
        # Clear existing data
        try:
            self.collection.delete(where={"source": {"$ne": ""}})
        except:
            # If collection is empty, just continue
            pass
        
        total_documents = 0
        
        # Process each file in the data directory
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            
            if filename.endswith('.json'):
                count = self._ingest_json_file(file_path, filename)
                total_documents += count
            elif filename.endswith('.md'):
                count = self._ingest_markdown_file(file_path, filename)
                total_documents += count
        
        return total_documents
    
    def _ingest_json_file(self, file_path: str, filename: str) -> int:
        """Ingest a JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of objects
            for i, item in enumerate(data):
                doc_id = f"{filename}_{i}"
                content = self._extract_content_from_json_item(item)
                
                if content:
                    documents.append(content)
                    metadatas.append({
                        "source": filename,
                        "doc_id": doc_id,
                        "type": "json"
                    })
                    ids.append(doc_id)
        elif isinstance(data, dict):
            # Single object or nested structure
            content = self._extract_content_from_json_item(data)
            if content:
                doc_id = f"{filename}_0"
                documents.append(content)
                metadatas.append({
                    "source": filename,
                    "doc_id": doc_id,
                    "type": "json"
                })
                ids.append(doc_id)
        
        # Add to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return len(documents)
    
    def _ingest_markdown_file(self, file_path: str, filename: str) -> int:
        """Ingest a Markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split markdown into chunks (by headers or sections)
        chunks = self._split_markdown_into_chunks(content)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc_id = f"{filename}_{i}"
                documents.append(chunk)
                metadatas.append({
                    "source": filename,
                    "doc_id": doc_id,
                    "type": "markdown"
                })
                ids.append(doc_id)
        
        # Add to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        return len(documents)
    
    def _extract_content_from_json_item(self, item: Any) -> str:
        """Extract text content from a JSON item"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            # Extract relevant text fields
            text_parts = []
            
            # Primary content fields (most important)
            if 'title' in item and item['title']:
                text_parts.append(f"Title: {item['title']}")
            
            if 'description' in item and item['description']:
                text_parts.append(f"Description: {item['description']}")
            
            # Special handling for transcriptText (most valuable for reviews)
            if 'transcriptText' in item and isinstance(item['transcriptText'], dict):
                transcript_content = item['transcriptText'].get('content', '')
                if transcript_content:
                    text_parts.append(f"Transcript: {transcript_content}")
            
            # Channel information (context)
            if 'channel_title' in item and item['channel_title']:
                text_parts.append(f"Channel: {item['channel_title']}")
            
            # Engagement metrics (popularity/authority indicators)
            if 'views' in item and item['views']:
                text_parts.append(f"Views: {item['views']}")
            
            if 'subscribers' in item and item['subscribers']:
                text_parts.append(f"Channel Subscribers: {item['subscribers']}")
            
            # Video metadata (context)
            if 'totalSeconds' in item and item['totalSeconds']:
                text_parts.append(f"Video Length: {item['totalSeconds']} seconds")
            
            if 'resolution' in item and item['resolution']:
                text_parts.append(f"Resolution: {item['resolution']}")
            
            if 'publishedAt' in item and item['publishedAt']:
                text_parts.append(f"Published: {item['publishedAt']}")
            
            # Include any other relevant string fields
            other_fields = ['brand', 'product', 'region']
            for field in other_fields:
                if field in item and item[field]:
                    text_parts.append(f"{field}: {item[field]}")
            
            return " ".join(text_parts)
        else:
            return str(item)
    
    def _split_markdown_into_chunks(self, content: str) -> List[str]:
        """Split markdown content into manageable chunks"""
        # Split by headers (lines starting with #)
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            if line.startswith('#') and current_chunk:
                # New header, save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Add the last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # If no headers found, split by paragraphs
        if len(chunks) <= 1:
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        
        return chunks

def main():
    """Main function to run ingestion"""
    ingester = DataIngester()
    total_docs = ingester.ingest_all_data()
    print(f"Successfully ingested {total_docs} documents!")

if __name__ == "__main__":
    main()
