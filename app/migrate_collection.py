#!/usr/bin/env python3
"""
Collection Migration Utility for BGE-M3 Transition

This script helps migrate from Nomic Embed Text v1 (768-dim) to BGE-M3 (1024-dim)
by creating a new collection and re-processing documents with the new embedding model.
"""

import os
import logging
import qdrant_client
from qdrant_client.models import Distance, VectorParams
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_collection_metadata(client, collection_name: str = "documents"):
    """Backup collection metadata before migration"""
    try:
        # Get all points with metadata
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Adjust based on your collection size
            with_payload=True,
            with_vectors=False  # We don't need vectors for metadata backup
        )
        
        metadata_backup = []
        for point in scroll_result[0]:
            metadata_backup.append({
                'id': point.id,
                'payload': point.payload
            })
        
        # Save to backup file
        backup_file = Path('/app/data/.backup/collection_metadata.json')
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(backup_file, 'w') as f:
            json.dump(metadata_backup, f, indent=2)
        
        logger.info(f"Backed up metadata for {len(metadata_backup)} documents to {backup_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error backing up collection metadata: {e}")
        return False

def create_bge_m3_collection(client, collection_name: str = "documents_bge_m3"):
    """Create a new collection optimized for BGE-M3 (1024 dimensions)"""
    try:
        # Check if collection already exists
        try:
            client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
            return True
        except Exception:
            pass
        
        # Create new collection with 1024 dimensions for BGE-M3
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        
        logger.info(f"Created new collection '{collection_name}' with 1024-dimensional vectors for BGE-M3")
        return True
        
    except Exception as e:
        logger.error(f"Error creating BGE-M3 collection: {e}")
        return False

def delete_old_collection(client, collection_name: str = "documents"):
    """Delete the old collection after successful migration"""
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted old collection '{collection_name}'")
        return True
    except Exception as e:
        logger.error(f"Error deleting old collection: {e}")
        return False

def rename_collection(client, old_name: str = "documents_bge_m3", new_name: str = "documents"):
    """Rename the new collection to the standard name"""
    try:
        # Qdrant doesn't have direct rename, so we need to:
        # 1. Create new collection with target name
        # 2. Copy all data
        # 3. Delete old collection
        
        # First, create the target collection
        try:
            collection_info = client.get_collection(old_name)
            vector_size = collection_info.config.params.vectors.size
            
            client.create_collection(
                collection_name=new_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            
            # Get all points from source collection
            scroll_result = client.scroll(
                collection_name=old_name,
                limit=10000,
                with_payload=True,
                with_vectors=True
            )
            
            if scroll_result[0]:
                # Prepare points for batch upload
                points = scroll_result[0]
                
                client.upsert(
                    collection_name=new_name,
                    points=points
                )
                
                logger.info(f"Copied {len(points)} points from '{old_name}' to '{new_name}'")
                
                # Delete old collection
                client.delete_collection(old_name)
                logger.info(f"Renamed collection '{old_name}' to '{new_name}'")
                return True
            else:
                logger.info("No points to copy during rename")
                return True
                
        except Exception as e:
            logger.error(f"Error during collection rename: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error renaming collection: {e}")
        return False

def main():
    """Main migration function"""
    logger.info("Starting BGE-M3 migration process...")
    
    # Connect to Qdrant
    client = qdrant_client.QdrantClient(
        host=os.environ.get("QDRANT_HOST", "qdrant"),
        port=6333
    )
    
    # Step 1: Backup existing collection metadata
    logger.info("Step 1: Backing up collection metadata...")
    if not backup_collection_metadata(client):
        logger.error("Failed to backup metadata. Aborting migration.")
        return False
    
    # Step 2: Create new BGE-M3 collection
    logger.info("Step 2: Creating BGE-M3 collection...")
    if not create_bge_m3_collection(client):
        logger.error("Failed to create BGE-M3 collection. Aborting migration.")
        return False
    
    logger.info("Migration setup complete!")
    logger.info("Next steps:")
    logger.info("1. Update your docker-compose.yml to use BGE-M3 configuration")
    logger.info("2. Restart services with: docker-compose -f docker-compose.bge-m3.yml up -d")
    logger.info("3. Re-process your documents to populate the new collection")
    logger.info("4. Run this script with --finalize to complete migration")
    
    return True

def finalize_migration():
    """Finalize migration by cleaning up old collection"""
    logger.info("Finalizing BGE-M3 migration...")
    
    client = qdrant_client.QdrantClient(
        host=os.environ.get("QDRANT_HOST", "qdrant"),
        port=6333
    )
    
    # Check if new collection has data
    try:
        collection_info = client.get_collection("documents_bge_m3")
        if collection_info.points_count > 0:
            logger.info(f"New collection has {collection_info.points_count} documents")
            
            # Rename new collection to standard name
            if rename_collection(client, "documents_bge_m3", "documents"):
                logger.info("Migration completed successfully!")
                return True
            else:
                logger.error("Failed to finalize migration")
                return False
        else:
            logger.error("New collection is empty. Please process documents first.")
            return False
            
    except Exception as e:
        logger.error(f"Error checking new collection: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize":
        finalize_migration()
    else:
        main()