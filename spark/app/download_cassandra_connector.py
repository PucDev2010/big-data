"""
Helper script to download Spark Cassandra Connector JAR for local PySpark usage
"""
import urllib.request
import os

def download_cassandra_connector():
    """Download Spark Cassandra Connector JAR file"""
    jar_url = "https://repo1.maven.org/maven2/com/datastax/spark/spark-cassandra-connector_2.12/3.5.0/spark-cassandra-connector_2.12-3.5.0.jar"
    jar_filename = "spark-cassandra-connector_2.12-3.5.0.jar"
    jar_dir = os.path.join(os.path.dirname(__file__), "jars")
    
    # Create jars directory if it doesn't exist
    os.makedirs(jar_dir, exist_ok=True)
    jar_path = os.path.join(jar_dir, jar_filename)
    
    if os.path.exists(jar_path):
        print(f"✓ JAR already exists: {jar_path}")
        return jar_path
    
    print(f"Downloading Spark Cassandra Connector from {jar_url}...")
    try:
        urllib.request.urlretrieve(jar_url, jar_path)
        print(f"✓ Successfully downloaded: {jar_path}")
        return jar_path
    except Exception as e:
        print(f"✗ Error downloading JAR: {e}")
        return None

if __name__ == "__main__":
    download_cassandra_connector()
