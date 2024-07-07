import pandas as pd
import json
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, LargeBinary, DateTime
from sqlalchemy.orm import sessionmaker
import datetime
import streamlit as st
import os

# Fetch database connection details from environment variables
db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME')

# Create the database connection URL
DATABASE_URL = f"postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Set up the database connection
engine = create_engine(DATABASE_URL)
metadata = MetaData()

class Dataset:
    def __init__(self):
        if 'datasets' not in metadata.tables:
            self.datasets = Table(
                'datasets', metadata,
                Column('id', Integer, primary_key=True),
                Column('name', String, nullable=False),
                Column('file_format', String, nullable=False),
                Column('file_size', Integer, nullable=False),
                Column('upload_date', DateTime, default=datetime.datetime.utcnow),
                Column('data', LargeBinary, nullable=False),
                extend_existing=True
            )
            metadata.create_all(engine)
        else:
            self.datasets = metadata.tables['datasets']
        self.Session = sessionmaker(bind=engine)

    def save_to_database(self, file_name, file_format, file_size, data):
        insert_statement = self.datasets.insert().values(
            name=file_name,
            file_format=file_format,
            file_size=file_size,
            data=data
        )
        with engine.connect() as conn:
            conn.execute(insert_statement)

    def fetch_datasets(self):
        with engine.connect() as conn:
            query = self.datasets.select().order_by(self.datasets.c.upload_date.desc())
            result = conn.execute(query)
            return result.fetchall()

    def dataset_exists(self, file_name):
        with engine.connect() as conn:
            query = self.datasets.select().where(self.datasets.c.name == file_name)
            result = conn.execute(query)
            return result.fetchone() is not None

    def delete_dataset(self, dataset_id):
        with engine.connect() as conn:
            delete_statement = self.datasets.delete().where(self.datasets.c.id == dataset_id)
            conn.execute(delete_statement)

    def get_dataset_by_id(self, dataset_id):
        with engine.connect() as conn:
            query = self.datasets.select().where(self.datasets.c.id == dataset_id)
            result = conn.execute(query)
            return result.fetchone()

    @staticmethod
    def try_parsing_csv(uploaded_file):
        try:
            return pd.read_csv(uploaded_file)
        except pd.errors.ParserError:
            try:
                return pd.read_csv(uploaded_file, delimiter=';')
            except pd.errors.ParserError:
                try:
                    return pd.read_csv(uploaded_file, delimiter='\t')
                except pd.errors.ParserError as e:
                    st.error(f"Error parsing the CSV file: {e}")
                    return None

    @staticmethod
    def try_parsing_json(uploaded_file):
        try:
            data = uploaded_file.read().decode('utf-8').strip().split('\n')
            uploaded_file.seek(0)
            json_data = [json.loads(line) for line in data if line.strip()]
            return pd.json_normalize(json_data)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing the JSON file: {e}")
            return None
        except Exception as e:
            st.error(f"Error parsing the JSON file: {e}")
            return None
