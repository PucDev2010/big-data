-- Create keyspace for job analytics
CREATE KEYSPACE IF NOT EXISTS job_analytics
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};

-- Use the keyspace
USE job_analytics;

-- Create table for job postings
CREATE TABLE IF NOT EXISTS job_postings (
    id UUID PRIMARY KEY,
    job_title TEXT,
    job_type TEXT,
    position_level TEXT,
    city TEXT,
    experience TEXT,
    skills TEXT,
    job_fields TEXT,
    salary TEXT,
    salary_min DOUBLE,
    salary_max DOUBLE,
    salary_avg DOUBLE,
    unit TEXT,
    event_time TIMESTAMP,
    event_type TEXT,
    exp_raw TEXT,
    exp_min_year DOUBLE,
    exp_max_year DOUBLE,
    exp_avg_year DOUBLE,
    exp_type TEXT
);

-- Cassandra Schema Creation Script
-- Run this script to create keyspace and tables for job postings

-- Create keyspace
CREATE KEYSPACE IF NOT EXISTS jobdb
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};

USE jobdb;

-- Create job_postings table
-- This table stores all job postings with their features
CREATE TABLE IF NOT EXISTS job_postings (
    id TEXT PRIMARY KEY,
    job_title TEXT,
    job_type TEXT,
    position_level TEXT,
    city TEXT,
    experience TEXT,
    skills TEXT,
    job_fields TEXT,
    salary TEXT,
    salary_min DOUBLE,
    salary_max DOUBLE,
    unit TEXT,
    avg_salary DOUBLE,
    salary_range DOUBLE,
    num_skills INT,
    num_fields INT,
    title_length INT,
    event_time TIMESTAMP,
    event_type TEXT,
    processed_at TIMESTAMP
);

-- Create index for common queries
CREATE INDEX IF NOT EXISTS idx_city ON job_postings (city);
CREATE INDEX IF NOT EXISTS idx_job_type ON job_postings (job_type);
CREATE INDEX IF NOT EXISTS idx_position_level ON job_postings (position_level);
CREATE INDEX IF NOT EXISTS idx_experience ON job_postings (experience);
CREATE INDEX IF NOT EXISTS idx_avg_salary ON job_postings (avg_salary);

-- Create materialized view for ML training queries
-- This view filters valid data for ML training
CREATE MATERIALIZED VIEW IF NOT EXISTS ml_training_data AS
SELECT 
    id,
    job_title,
    job_type,
    position_level,
    city,
    experience,
    skills,
    job_fields,
    salary_min,
    salary_max,
    avg_salary,
    num_skills,
    num_fields,
    title_length,
    event_time
FROM job_postings
WHERE avg_salary > 0 
    AND salary_min > 0 
    AND salary_max > 0
PRIMARY KEY (id, avg_salary);

-- Create table for storing ML model metadata
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

-- Create table for ML predictions
CREATE TABLE IF NOT EXISTS ml_predictions (
    prediction_id UUID PRIMARY KEY,
    job_id TEXT,
    model_id UUID,
    predicted_salary DOUBLE,
    actual_salary DOUBLE,
    prediction_date TIMESTAMP,
    features MAP<TEXT, TEXT>
);
