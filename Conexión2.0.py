import psycopg2

psw='navidadp3'

engine = psycopg2.connect(
dbname="world",
user="postgres",
password=psw,
host="p3.curxufagptbe.us-east-1.rds.amazonaws.com",
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


