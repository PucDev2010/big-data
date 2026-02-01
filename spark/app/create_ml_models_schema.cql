-- Create keyspace for ML models
CREATE KEYSPACE IF NOT EXISTS jobdb
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};

-- Use the keyspace
USE jobdb;

-- Create table for ML model metadata
CREATE TABLE IF NOT EXISTS ml_models (
    model_id UUID PRIMARY KEY,
    model_name TEXT,
    model_type TEXT,
    training_date TIMESTAMP,
    accuracy DOUBLE,
    mae DOUBLE,
    rmse DOUBLE,
    r2_score DOUBLE,
    feature_columns LIST<TEXT>,
    model_path TEXT,
    version INT
);

-- Create index on training_date for faster queries
CREATE INDEX IF NOT EXISTS idx_training_date ON ml_models (training_date);
