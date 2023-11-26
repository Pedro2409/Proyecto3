import psycopg2
engine = psycopg2.connect(
dbname="p3",
user="postgres",
password="proyecto3",
host="databasep3.cut51ddih3ok.us-east-1.rds.amazonaws.com",
port='5432'
)

cursor = engine.cursor()

query = """
SELECT * 
FROM pg_catalog.pg_tables 
WHERE schemaname='public';"""
cursor.execute(query)
result = cursor.fetchall()
result

query = """
SELECT * 
FROM dataP3
LIMIT 10;"""
cursor.execute(query)
result = cursor.fetchall()
result

