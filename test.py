from dotenv import load_dotenv
import os
import dspy
import datetime
from langchain_community.utilities import SQLDatabase
from difflib import get_close_matches

# Load environment variables
load_dotenv()

# Get API key from .env file
api_key = os.getenv("OPENAI_API_KEY")

# Configure dspy with GPT-4o-mini model
dspy.configure(
    lm=dspy.OpenAI(model="gpt-4o-mini", api_key=api_key, max_tokens=500)
)

# Load the SQLite temp database
db = SQLDatabase.from_uri('sqlite:///temp_db.db')

# Define the table identifier signature
class TableIdentifier(dspy.Signature):
    """Identify table names and group by column names."""
    
    user_input = dspy.InputField()
    table_names = dspy.OutputField(desc="List of identified table names")
    group_by_columns = dspy.OutputField(desc="List of group by column names")

# Create the table identifier module
class TableIdentifierModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.identify_tables = dspy.Predict(TableIdentifier)

    def forward(self, user_input):
        result = self.identify_tables(user_input=user_input)
        return result.table_names, result.group_by_columns

# Main function to run the table identifier
def main():
    identifier = TableIdentifierModule()
    
    existing_table_names = db.get_usable_table_names()  # Fetching results from the cursor
    print(existing_table_names)

    # Get user input
    user_input = input("Enter your input to identify table names and group by column names: ")
    
    # Identify table names and group by column names
    table_names, group_by_columns = identifier(user_input)
    
    # Check if the identified tables exist in the database
    # cursor = db._execute("SELECT name FROM sqlite_master WHERE type='table';")  # Ensure this returns a cursor
    # existing_table_names = [table[0] for table in existing_tables]
    
    # Find the closest matching table names
    # closest_matches = get_close_matches(table_names, existing_table_names, n=3, cutoff=0.5)
    closest_matches = table_names
    print(closest_matches)
    if not closest_matches:
        print(f"Error: No similar table names found in the database for the input: {user_input}")
    else:
        # Pass the list of table names and the database into the prompt
        prompt = f"Identified Table Names: {closest_matches}\nIdentified Group By Column Names: {group_by_columns}\nDatabase Tables: {existing_table_names}"
        print(prompt)
        
        # Generate timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"identified_tables_{timestamp}.txt"

        # Save the output to a text file with timestamp
        # with open(filename, "w") as file:
        #     file.write("Identified Table Names:\n")
        #     for table_name in table_names:
        #         file.write(f"- {table_name}\n")
        #     file.write("\nIdentified Group By Column Names:\n")
        #     for column_name in group_by_columns:
        #         file.write(f"- {column_name}\n")
        
        # print(f"\nOutput has been saved to '{filename}'")
    
if __name__ == "__main__":
    main()
