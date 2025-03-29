import pymongo
from pymongo import MongoClient
import pprint
import os
from dotenv import load_dotenv

def main():
      # Load environment variables from .env file
    load_dotenv()
    
    # Get connection string from environment variable
    connection_string = os.getenv("MONGODB_CONNECTION_STRING")

    try:
        # Connect to MongoDB
        client = MongoClient(connection_string)
        
        # Access the sample_mflix database
        db = client.sample_mflix
        
        # Access the movies collection
        movies_collection = db.movies
        
        # Find the first 10 movies
        movies = movies_collection.find().limit(10)
        
        # Print each movie
        print("First 10 movies from the collection:")
        for i, movie in enumerate(movies, 1):
            print(f"\nMovie {i}:")
            pprint.pprint(movie)
        
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the connection
        if 'client' in locals():
            client.close()
            print("\nMongoDB connection closed.")

if __name__ == "__main__":
    main()
