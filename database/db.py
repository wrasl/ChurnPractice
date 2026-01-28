import sqlite3

class CustomerDB:

    def __init__(self, db_path="customers.db"):

        self.db_path = db_path
        self.conn = None
        self.cursor = None


    def __enter__(self):

        self.connect()
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()


    def connect(self):

        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise


    def close(self):

        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None


    def commit(self):

        if not self.conn:
            raise RuntimeError("Database connection is not established.")
        
        self.conn.commit()
