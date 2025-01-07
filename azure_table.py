from azure.data.tables import TableServiceClient, TableEntity
from azure.core.exceptions import ResourceExistsError
import uuid
import datetime
from datetime import datetime
import os
import re
import json
from collections import Counter
import pytz  # If you want to handle timezone-aware timestamps
from dotenv import load_dotenv

load_dotenv()

# Define your Azure Table connection string and table name 
CONNECTION_STRING = os.getenv("AZURE_TABLE_CONNECTIONSTRING")

# def getalltablesname():
#     tablename = []
#     service_client = TableServiceClient.from_connection_string(conn_str=CONNECTION_STRING)
#     tables = service_client.list_tables()
#     for table in tables: 
#         tablename.append(table.name)
#         #print(table.name)
#     return tablename


# Initialize TableServiceClient
def get_table_client(TABLE_NAME):
    service_client = TableServiceClient.from_connection_string(conn_str=CONNECTION_STRING)
    try:
        # Create the table if it doesn't exist
        service_client.create_table(TABLE_NAME)
        print(f"Table '{TABLE_NAME}' created successfully.")
    except ResourceExistsError:
        print(f"Table '{TABLE_NAME}' already exists.")
    return service_client.get_table_client(TABLE_NAME)

# Insert a record into the table
def insert_data_graphtable(metadata,sheet_name,name):
    try:
        table_client = get_table_client("GraphData")

        # Entity must have PartitionKey and RowKey
        entity = {
            "PartitionKey": "ExcelSheet",  # Logical grouping of rows
            "RowKey": str(uuid.uuid4()),  # Unique within PartitionKey
            "sheetname": sheet_name,
            "metadata": metadata,
            "name": name,
            "Date":  datetime.now()
        }

        # Insert or merge the entity
        table_client.create_entity(entity)
        print("Data inserted successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

## INSERT DATA

def sanitize_property_name(name):
    # Replace spaces with underscores and remove any non-alphanumeric characters
    sanitized_name = re.sub(r'[^A-Za-z0-9_]', '', name)
    return sanitized_name

# Insert a list of JSON data with sanitized property names
def insert_json_data_list(json_data_list, table_name):
    table_client = get_table_client(table_name)

    for index, json_data in enumerate(json_data_list):
        # Ensure json_data is a dictionary (in case it's a string)
        if isinstance(json_data, str):
            json_data = json.loads(json_data)  # Convert from JSON string to dictionary

        # Construct a unique PartitionKey and RowKey for each entry
        partition_key = f"Partition{index + 1}"  # You can modify this to your logic
        row_key = str(index + 1)  # Ensure RowKey is unique, use the index for simplicity

        # Construct the entity
        entity = {
            "PartitionKey": partition_key,
            "RowKey": row_key,
        }

        # Add JSON fields to the entity with sanitized property names
        for key, value in json_data.items():
            # Sanitize the property name
            sanitized_key = sanitize_property_name(key)

            # Handle datetime fields: convert datetime to string (ISO format)
            if isinstance(value, datetime):
                # If it's naive datetime, localize it to a timezone (e.g., UTC)
                if value.tzinfo is None:
                    value = pytz.utc.localize(value)
                entity[sanitized_key] = value.isoformat()  # Convert to ISO string
            # Handle list fields: convert lists to JSON strings for storage
            elif isinstance(value, list):
                entity[sanitized_key] = json.dumps(value)  # Use json.dumps to ensure valid JSON format
            else:
                entity[sanitized_key] = value

        # Insert entity into the table
        try:
            table_client.create_entity(entity=entity)
            print(f"JSON data for RowKey '{row_key}' inserted successfully.")
        except Exception as e:
            print(f"Failed to insert data for RowKey '{row_key}': {e}")


######################################################
#get all data from graphData table from azure.
def getGraphData():
    try:
        # Get the table client
        table_client = get_table_client("GraphData") 
        # Query all entities in the table
        all_records = list(table_client.list_entities())
        # entities = table_client.query_entities(
        #     query_filter="",
        #     select=['tablename']  # Specify only the column you need
        # )      
        return all_records
    
    except HttpResponseError as e:
        print(f"An error occurred: {e}")
        return None
####################################################
#get all name from graphData table azure
def getAllGraphName():
    # Query to select specific columns
    table_client = get_table_client("GraphData")
    selected_columns = ["name"]

    # Fetch the data
    name=[]
    try:
        entities = table_client.query_entities(select=selected_columns,query_filter='')
        for entity in entities:
            name.append(entity.get('name'))
            #print(f"Name: {entity.get('name')}")
        return name
    except Exception as e:
        print(f"An error occurred: {e}")
###################################################
#####################################################
## get all column name from azure table
def getallcolumnname(tablename):
    table_client = get_table_client(tablename)
    column_names = set()  # Using a set to avoid duplicates
    try:
        # Fetch a few entities to get properties (columns)
        entities = table_client.list_entities()  # Limiting to default 100 entities
        for entity in entities:
            # Exclude PartitionKey, RowKey, and Timestamp
            for key in entity.keys():
                if key not in ['PartitionKey', 'RowKey', 'Timestamp']:
                    column_names.add(key)  # Collect column names dynamically
    except Exception as e:
        print(f"Error fetching column names from {tablename}: {e}")
    
    return list(column_names)

# Function to fetch data from a table with selected columns
def fetch_table_data(table_name, select_columns, query_filter=None):
    table_client = get_table_client(table_name)
    print("----------------tablename--------------------")
    print(table_name)
    try:
        # Fetch data with optional filter
        entities = table_client.query_entities(select=select_columns, query_filter=query_filter)
        results = []
        for entity in entities:
            results.append({col: entity.get(col) for col in select_columns})
        return results
    except Exception as e:
        print(f"Error fetching data from table {table_name}: {e}")
        return []

# Function to get metadata and keywords data
def get_TableData(tablename):

    table1_name = tablename
    table2_name = f"{tablename}sourcenode"

    # # Fetch columns dynamically from both tables
    table2_columns = getallcolumnname(table2_name)  # Get dynamic columns for the second table
    keywords_data = fetch_table_data(table2_name, table2_columns)

    return keywords_data

#########################################
##############################################
def transform_data(input_data):
    nodes = []
    links = []
    
    for entry in input_data:
        # Parse the JSON-like strings into actual lists
        group = json.loads(entry['group'])
        label = json.loads(entry['label'])
        nodeid = json.loads(entry['nodeid'])
        relationship = json.loads(entry['relationship'])
        source = json.loads(entry['source'])
        target = json.loads(entry['target'])
        
        # Construct nodes
        for i in range(len(nodeid)):
            nodes.append({
                "id": nodeid[i],
                "label": label[i],
                "group": group[i]
            })
        
        # Construct links
        for i in range(len(source)):
            links.append({
                "source": source[i],
                "target": target[i],
                "relationship": relationship[i]
            })
    
    return {
        "node": nodes,
        "link": links
    }
def get_networkgraphdata_by_id(tablename):
    table1_name = tablename
    table2_name = f"{tablename}sourcenode"
    # # Fetch columns dynamically from both tables
    table2_columns = ["nodeid","label","relationship","source","target","group"]
    network_data = fetch_table_data(table2_name, table2_columns)

    return transform_data(network_data)

#########################################
from collections import Counter

def get_word_counts(api_data):
    # Initialize a Counter to accumulate counts
    total_counts = Counter()
    
    # Iterate over the "data" values in the API response
    for value in api_data:
        # Convert the string representation of the list to an actual list
        items = json.loads(value)
        # Update the Counter with the items in the list
        total_counts.update(items)
    
    # Convert the Counter object to a dictionary
    return dict(total_counts)

def get_wordclusterdata(tablename):
    table1_name = tablename
    table2_name = f"{tablename}sourcenode"
    
    # Fetch only the "keywords" column from the table
    table2_columns = ["keywords"]
    network_data = fetch_table_data(table2_name, table2_columns)  # List of dictionaries

    # Extract keywords into a flat list
    keywords_list = []
    for row in network_data:
        # Append "keywords" field from each dictionary, if it exists
        if "keywords" in row:
            keywords = row["keywords"]
            if isinstance(keywords, list):
                keywords_list.extend(keywords)  # If it's a list, extend the list
            elif isinstance(keywords, str):
                keywords_list.append(keywords)  # If it's a string, append directly
    
    # Generate the word cluster data
    output_object = get_word_counts(keywords_list)
    return output_object

#####################################################
def get_reinforcementlearn_data(tablename):
    table1_name = tablename
    table2_name = f"{tablename}sourcenode"
    
    # Fetch only the "keywords" column from the table
    table2_columns = ["metadata"]
    network_data = fetch_table_data(table2_name, table2_columns)  # List of dictionaries
    #print(network_data)
    return network_data