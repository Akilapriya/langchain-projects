import sqlite3

# connect to sqlite
connection = sqlite3.connect("student.db")

# create a cursor opbject to insert record, create task
cursor = connection.cursor()

# create the table
table_info = """
create table STUDENT (NAME VARCHAR(25), CLASS VARCHAR(25), SECTION VARCHAR(25), MARKS INT)
"""
cursor.execute(table_info)

# insert some more records
(cursor.execute(""" Insert Into STUDENT values("Krish, "DATASCIENCE, "B", 90)"""),)
(cursor.execute(""" Insert Into STUDENT values("Pri, "DATASCIENCE, "B", 87)"""),)
(cursor.execute(""" Insert Into STUDENT values("Aki, "DEVOPS, "A", 67)"""),)
(cursor.execute(""" Insert Into STUDENT values("John, "DATASCIENCE, "B", 70)"""),)
(cursor.execute(""" Insert Into STUDENT values("Jacab, "DEVOPS, "C", 88)"""),)
(cursor.execute(""" Insert Into STUDENT values("Mahi, "DATASCIENCE, "A", 56)"""),)

# display all records
print("The inserted records are")
data = cursor.execute("""select * from STUDENT""")
for row in data:
    print(row)

# commit your chnges in database
connection.commit()
connection.close()
