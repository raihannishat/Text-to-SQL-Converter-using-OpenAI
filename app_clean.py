import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv('config.env')

class PostgreSQLManager:
    def __init__(self):
        self.engine = None
        self.inspector = None
        
    def connect(self, host, port, database, user, password):
        try:
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_string)
            self.inspector = inspect(self.engine)
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
            return False
    
    def get_column_descriptions(self, schema="public"):
        if not self.inspector:
            return {}
        descriptions = {}
        try:
            table_names = self.inspector.get_table_names(schema=schema)
            for table_name in table_names:
                columns = self.inspector.get_columns(table_name, schema=schema)
                table_descriptions = {}
                for column in columns:
                    col_name = column['name']
                    col_type = str(column['type'])
                    is_pk = column.get('primary_key', False)
                    is_nullable = column.get('nullable', True)
                    default = column.get('default', None)
                    comment = column.get('comment', '')
                    desc = f"Type: {col_type}; Nullable: {is_nullable}; Default: {default}; PK: {is_pk}; Comment: {comment}"
                    # Add sample values
                    try:
                        values = self.get_distinct_values(table_name, col_name, schema, 5)
                        if values:
                            desc += f"; Example values: {', '.join(map(str, values))}"
                    except:
                        pass
                    table_descriptions[col_name] = desc
                descriptions[table_name] = table_descriptions
        except Exception as e:
            st.error(f"Error getting column descriptions: {str(e)}")
        return descriptions
    
    def get_distinct_values(self, table_name, column_name, schema="public", limit=10):
        """Get distinct values for a specific column to help with query generation"""
        if not self.engine:
            return []
        
        try:
            query = f"SELECT DISTINCT {column_name} FROM {schema}.{table_name} WHERE {column_name} IS NOT NULL LIMIT {limit}"
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                values = [row[0] for row in result.fetchall()]
                return values
        except Exception as e:
            return []
    
    def get_foreign_key_relations(self, schema="public"):
        """Get foreign key relationships between tables"""
        if not self.inspector:
            return {}
        
        relations = {}
        try:
            table_names = self.inspector.get_table_names(schema=schema)
            for table_name in table_names:
                foreign_keys = self.inspector.get_foreign_keys(table_name, schema=schema)
                detailed_fks = []
                for fk in foreign_keys:
                    detailed_fks.append({
                        'constrained_columns': fk.get('constrained_columns', []),
                        'referred_table': fk.get('referred_table', ''),
                        'referred_columns': fk.get('referred_columns', []),
                        'name': fk.get('name', ''),
                        'options': fk.get('options', {}),
                        'onupdate': fk.get('onupdate', ''),
                        'ondelete': fk.get('ondelete', ''),
                        'business': f"{table_name}.{fk.get('constrained_columns', [''])[0]} references {fk.get('referred_table', '')}.{fk.get('referred_columns', [''])[0]} (business meaning: {fk.get('name', '')})",
                        'example_join': f"SELECT ... FROM {table_name} JOIN {fk.get('referred_table', '')} ON {table_name}.{fk.get('constrained_columns', [''])[0]} = {fk.get('referred_table', '')}.{fk.get('referred_columns', [''])[0]}"
                    })
                if detailed_fks:
                    relations[table_name] = detailed_fks
        except Exception as e:
            st.error(f"Error getting foreign key relations: {str(e)}")
        return relations
    
    def get_table_info(self, schema="public"):
        if not self.inspector:
            return {}
        schema_info = {}
        try:
            table_names = self.inspector.get_table_names(schema=schema)
            for table_name in table_names:
                columns = []
                for column in self.inspector.get_columns(table_name, schema=schema):
                    col_info = {
                        'name': column['name'],
                        'type': str(column['type']),
                        'nullable': column.get('nullable', True),
                        'default': column.get('default', None),
                        'primary_key': column.get('primary_key', False),
                        'autoincrement': column.get('autoincrement', False),
                        'comment': column.get('comment', ''),
                        'unique': False,  # will update below
                        'indexed': False, # will update below
                    }
                    columns.append(col_info)
                # Add unique/index info
                indexes = self.inspector.get_indexes(table_name, schema=schema)
                uniques = set()
                indexed = set()
                for idx in indexes:
                    for col in idx.get('column_names', []):
                        indexed.add(col)
                        if idx.get('unique', False):
                            uniques.add(col)
                for col in columns:
                    if col['name'] in uniques:
                        col['unique'] = True
                    if col['name'] in indexed:
                        col['indexed'] = True
                schema_info[table_name] = columns
        except Exception as e:
            st.error(f"Error getting table info: {str(e)}")
        return schema_info
    
    def get_sample_data(self, table_name, limit=1, schema="public"):
        if not self.engine:
            return []
        
        try:
            query = f"SELECT * FROM {schema}.{table_name} LIMIT {limit}"
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result.fetchall()]
                return data
        except Exception as e:
            return []
    
    def execute_query(self, sql_query):
        if not self.engine:
            return []
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result.fetchall()]
                return data
        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")
            return []

def initialize_openai():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == "your_openai_api_key_here":
        st.error("❌ OpenAI API key not found in config.env")
        return None
    
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=api_key
    )

def create_sql_prompt():
    return PromptTemplate(
        input_variables=["schema", "sample_data", "query", "relations", "values", "descriptions"],
        template="""
You are an expert SQL developer. Convert the natural language query to PostgreSQL SQL.

IMPORTANT: Use ALL the detailed schema, column descriptions, relationships, and sample data provided below. Do NOT make any assumptions about tables, columns, or relationships that are not explicitly described.

### Database Schema (ALL Tables, Columns, Types, Constraints, Indexes, Defaults, Nullability, PK/FK, Unique, etc.):
{schema}

### Column Descriptions (Business context, usage, allowed values, units, examples, data quality notes):
{descriptions}

### Entity Relations (Source/target, type, on delete/update, business meaning, example join):
{relations}

### Sample Data/Values (From ALL tables, including min/max, unique, nullability, examples):
{sample_data}

### Query:
{query}

### Instructions:
- Use ONLY the tables, columns, and relationships described above.
- Use the column descriptions to understand the business meaning, allowed values, and usage of each field.
- Use the actual column values and sample data to understand valid values and data patterns.
- Use foreign key relationships to create JOINs only when necessary and described above.
- For mapping/status/flag queries, use the relevant boolean or status column directly (e.g., is_mapped_with_user_village_mapping, status).
- Do NOT create JOINs unless the mapping or relationship is explicitly described in the schema and relationships.
- For boolean columns, use TRUE/FALSE in WHERE clause as shown in the column description.
- For status columns, use the actual status values shown in the column values.
- For reference columns (e.g., *_id), use them to JOIN only if a foreign key relationship is described.
- If a column description says "use this column directly" or "do not use JOINs", strictly avoid JOINs for that logic and use the column in the WHERE clause.
- If a boolean column (like is_mapped_with_user_village_mapping) directly indicates mapping, use it in the WHERE clause. Do NOT use JOINs for mapping logic if the column description says so.
- Generate only the SQL query without explanation.
- Use proper PostgreSQL syntax.

#### Example:
-- To get all designations where mapping is true:
SELECT * FROM designation WHERE is_mapped_with_user_village_mapping = TRUE;

-- To get all active projects:
SELECT * FROM project WHERE status = 'Active';

Generate only the SQL query without explanation. Use proper PostgreSQL syntax.
"""
    )

def generate_sql(llm, schema, sample_data, query, relations=None, db_manager=None, db_schema="public"):
    try:
        # Get complete schema information for ALL tables
        complete_schema_text = ""
        complete_values_text = ""
        complete_relations_text = ""
        complete_sample_data_text = ""
        complete_descriptions_text = ""
        
        if db_manager:
            # Get all tables and their complete information
            schema_info = db_manager.get_table_info(db_schema)
            
            st.info(f"📊 Found {len(schema_info)} tables in database")
            
            # Get column descriptions
            descriptions = db_manager.get_column_descriptions(db_schema)
            
            for table_name, columns in schema_info.items():
                # Add table header
                complete_schema_text += f"\n{'='*50}\n"
                complete_schema_text += f"Table: {table_name} ({len(columns)} columns)\n"
                complete_schema_text += f"{'='*50}\n"
                
                # Add all columns with details
                for column in columns:
                    pk = " (PK)" if column['primary_key'] else ""
                    nullable = " (NULL)" if column.get('nullable', True) else " (NOT NULL)"
                    complete_schema_text += f"  - {column['name']}: {column['type']}{pk}{nullable}\n"
                
                # Add column descriptions for this table
                if table_name in descriptions:
                    complete_descriptions_text += f"\n{table_name} Column Descriptions:\n"
                    complete_descriptions_text += "="*50 + "\n"
                    for col_name, description in descriptions[table_name].items():
                        complete_descriptions_text += f"  {col_name}: {description}\n"
                
                # Get sample data for this table
                sample_data = db_manager.get_sample_data(table_name, 5, db_schema)
                if sample_data:
                    complete_sample_data_text += f"\n{table_name} sample data:\n"
                    for i, row in enumerate(sample_data, 1):
                        complete_sample_data_text += f"  Row {i}: {row}\n"
                
                # Get distinct values for ALL columns (not just important ones)
                for col in columns:
                    col_name = col['name']
                    try:
                        values = db_manager.get_distinct_values(table_name, col_name, db_schema, 15)
                        if values and len(values) > 0:
                            # Only show if there are meaningful values (not all NULL)
                            non_null_values = [v for v in values if v is not None and str(v).strip()]
                            if non_null_values:
                                complete_values_text += f"{table_name}.{col_name} values: {', '.join(map(str, non_null_values[:10]))}\n"
                    except Exception as e:
                        # Skip columns that can't be queried
                        continue
            
            # Get all foreign key relationships
            if relations:
                complete_relations_text = "\nForeign Key Relationships:\n"
                complete_relations_text += "="*50 + "\n"
                for table_name, foreign_keys in relations.items():
                    for fk in foreign_keys:
                        from_col = fk.get('constrained_columns', [''])[0]
                        to_table = fk.get('referred_table', '')
                        to_col = fk.get('referred_columns', [''])[0]
                        constraint_name = fk.get('name', '')
                        complete_relations_text += f"  {table_name}.{from_col} → {to_table}.{to_col} (Constraint: {constraint_name})\n"
            else:
                complete_relations_text = "\nNo foreign key relationships found in database.\n"
        
        # Increase limits to provide more comprehensive information
        max_schema_length = 4000
        max_values_length = 1500
        max_relations_length = 800
        max_sample_length = 1500
        max_descriptions_length = 2000
        
        if len(complete_schema_text) > max_schema_length:
            complete_schema_text = complete_schema_text[:max_schema_length] + "\n... (schema truncated)"
        
        if len(complete_values_text) > max_values_length:
            complete_values_text = complete_values_text[:max_values_length] + "\n... (values truncated)"
        
        if len(complete_relations_text) > max_relations_length:
            complete_relations_text = complete_relations_text[:max_relations_length] + "\n... (relations truncated)"
        
        if len(complete_sample_data_text) > max_sample_length:
            complete_sample_data_text = complete_sample_data_text[:max_sample_length] + "\n... (sample data truncated)"
        
        if len(complete_descriptions_text) > max_descriptions_length:
            complete_descriptions_text = complete_descriptions_text[:max_descriptions_length] + "\n... (descriptions truncated)"
        
        prompt = create_sql_prompt()
        chain = prompt | llm
        
        result = chain.invoke({
            "schema": complete_schema_text,
            "sample_data": complete_sample_data_text,
            "query": query,
            "relations": complete_relations_text,
            "values": complete_values_text,
            "descriptions": complete_descriptions_text
        })
        
        sql = result.content.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.endswith("```"):
            sql = sql[:-3]
        return sql.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Text-to-SQL Converter using OpenAI",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Text-to-SQL Converter using OpenAI")
    st.markdown("Transform natural language queries into PostgreSQL SQL using GPT-4o-mini")
    
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"
    
    # Sidebar
    with st.sidebar:
        st.header("🔗 PostgreSQL Connection")
        
        # API Key Status
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != "your_openai_api_key_here":
            st.success("✅ OpenAI API Key loaded")
        else:
            st.warning("⚠️ OpenAI API Key not found")
        
        # Connection inputs
        db_host = st.text_input("Host", value="localhost")
        db_port = st.text_input("Port", value="5432")
        db_name = st.text_input("Database", value="postgres")
        db_user = st.text_input("Username", value="postgres")
        db_password = st.text_input("Password", type="password")
        db_schema = st.text_input("Schema", value="public")
        
        if st.button("🔗 Connect", type="primary"):
            if db_host and db_name and db_user and db_password:
                with st.spinner("Connecting..."):
                    db_manager = PostgreSQLManager()
                    if db_manager.connect(db_host, db_port, db_name, db_user, db_password):
                        st.session_state.db_manager = db_manager
                        st.session_state.db_schema = db_schema
                        st.success("✅ Connected!")
                    else:
                        st.error("❌ Connection failed")
            else:
                st.error("Please fill all required fields")
        
        if st.session_state.db_manager:
            st.success("✅ Database Connected")
            
            # Navigation
            st.header("📄 Navigation")
            if st.button("🏠 Main Page"):
                st.session_state.current_page = "main"
            if st.button("📊 Schema Info"):
                st.session_state.current_page = "schema"
            if st.button("🔗 Entity Relations"):
                st.session_state.current_page = "relations"
            if st.button("📝 Column Descriptions"):
                st.session_state.current_page = "descriptions"
    
    # Main content
    if st.session_state.db_manager:
        db_manager = st.session_state.db_manager
        db_schema = st.session_state.get('db_schema', 'public')
        
        if st.session_state.current_page == "schema":
            # Schema Info Page
            st.header("📊 Database Schema Information")
            
            # Database Info
            try:
                with db_manager.engine.connect() as conn:
                    result = conn.execute(text("SELECT version()"))
                    version = result.fetchone()[0]
                    
                    result = conn.execute(text("SELECT current_database()"))
                    db_name = result.fetchone()[0]
                    
                    result = conn.execute(text("SELECT current_user"))
                    current_user = result.fetchone()[0]
                    
                    result = conn.execute(text("SELECT current_schema"))
                    current_schema = result.fetchone()[0]
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Database", db_name)
                    with col2:
                        st.metric("User", current_user)
                    with col3:
                        st.metric("Schema", current_schema)
                    with col4:
                        st.metric("Version", version.split()[0])
                    with col5:
                        st.metric("Status", "Connected")
            except Exception as e:
                st.error(f"Error getting database info: {str(e)}")
            
            # Table Information
            st.subheader("📋 Table Details")
            schema_info = db_manager.get_table_info(db_schema)
            
            if schema_info:
                for table_name, columns in schema_info.items():
                    with st.expander(f"📋 {table_name} ({len(columns)} columns)"):
                        # Create a DataFrame for better display
                        table_data = []
                        for col in columns:
                            table_data.append({
                                'Column': col['name'],
                                'Type': col['type'],
                                'Primary Key': '✓' if col['primary_key'] else ''
                            })
                        
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Sample data
                        if st.button(f"Show Sample Data for {table_name}"):
                            sample_data = db_manager.get_sample_data(table_name, 5, db_schema)
                            if sample_data:
                                st.subheader(f"Sample Data from {table_name}")
                                sample_df = pd.DataFrame(sample_data)
                                st.dataframe(sample_df, use_container_width=True)
                            else:
                                st.info(f"No data available in {table_name}")
            else:
                st.warning("No tables found in database")
        elif st.session_state.current_page == "relations":
            # Entity Relations Page
            st.header("🔗 Database Entity Relations")
            
            # Get foreign key relationships
            relations = db_manager.get_foreign_key_relations(db_schema)
            
            if relations:
                st.subheader("📋 Foreign Key Relationships")
                
                # Create a summary of all relationships
                all_relations = []
                for table_name, foreign_keys in relations.items():
                    for fk in foreign_keys:
                        all_relations.append({
                            'From Table': table_name,
                            'From Column': fk.get('constrained_columns', [''])[0],
                            'To Table': fk.get('referred_table', ''),
                            'To Column': fk.get('referred_columns', [''])[0],
                            'Constraint Name': fk.get('name', '')
                        })
                
                if all_relations:
                    # Display relationships in a table
                    relations_df = pd.DataFrame(all_relations)
                    st.dataframe(relations_df, use_container_width=True)
                    
                    # Show relationship diagram
                    st.subheader("🎯 Relationship Summary")
                    for relation in all_relations:
                        st.info(f"**{relation['From Table']}.{relation['From Column']}** → **{relation['To Table']}.{relation['To Column']}**")
                    
                    # Show detailed relationships by table
                    st.subheader("📋 Detailed Relationships by Table")
                    for table_name, foreign_keys in relations.items():
                        with st.expander(f"🔗 {table_name} Relationships"):
                            for fk in foreign_keys:
                                from_col = fk.get('constrained_columns', [''])[0]
                                to_table = fk.get('referred_table', '')
                                to_col = fk.get('referred_columns', [''])[0]
                                
                                st.write(f"**{from_col}** → **{to_table}.{to_col}**")
                                
                                # Show sample data for this relationship
                                if st.button(f"Show {table_name} → {to_table} data", key=f"rel_{table_name}_{to_table}"):
                                    try:
                                        # Get sample data showing the relationship
                                        query = f"""
                                        SELECT t1.*, t2.* 
                                        FROM {db_schema}.{table_name} t1 
                                        LEFT JOIN {db_schema}.{to_table} t2 
                                        ON t1.{from_col} = t2.{to_col} 
                                        LIMIT 5
                                        """
                                        sample_data = db_manager.execute_query(query)
                                        if sample_data:
                                            st.subheader(f"Sample {table_name} → {to_table} Data")
                                            sample_df = pd.DataFrame(sample_data)
                                            st.dataframe(sample_df, use_container_width=True)
                                        else:
                                            st.info(f"No relationship data found between {table_name} and {to_table}")
                                    except Exception as e:
                                        st.error(f"Error showing relationship data: {str(e)}")
                else:
                    st.warning("No foreign key relationships found")
            else:
                st.warning("No foreign key relationships found in database")
                
                # Show table relationships summary
                st.subheader("📊 Table Summary")
                schema_info = db_manager.get_table_info(db_schema)
                if schema_info:
                    table_summary = []
                    for table_name, columns in schema_info.items():
                        primary_keys = [col['name'] for col in columns if col['primary_key']]
                        foreign_keys = [col['name'] for col in columns if 'foreign' in col['type'].lower() or 'fk' in col['name'].lower()]
                        
                        table_summary.append({
                            'Table': table_name,
                            'Columns': len(columns),
                            'Primary Keys': ', '.join(primary_keys) if primary_keys else 'None',
                            'Foreign Keys': ', '.join(foreign_keys) if foreign_keys else 'None'
                        })
                    
                    summary_df = pd.DataFrame(table_summary)
                    st.dataframe(summary_df, use_container_width=True)
        elif st.session_state.current_page == "descriptions":
            # Column Descriptions Page
            st.header("📝 Column Descriptions")
            st.markdown("Detailed descriptions of what each column does and why it's used")
            
            # Get column descriptions for all tables
            descriptions = db_manager.get_column_descriptions(db_schema)
            
            if descriptions:
                for table_name, table_descriptions in descriptions.items():
                    with st.expander(f"📋 {table_name} - Column Descriptions", expanded=True):
                        st.subheader(f"Table: {table_name}")
                        
                        # Create a DataFrame for better display
                        desc_data = []
                        for col_name, description in table_descriptions.items():
                            desc_data.append({
                                'Column': col_name,
                                'Description': description
                            })
                        
                        if desc_data:
                            df = pd.DataFrame(desc_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Also show in a more readable format
                            st.markdown("**Detailed Descriptions:**")
                            for col_name, description in table_descriptions.items():
                                st.markdown(f"• **{col_name}**: {description}")
                        else:
                            st.info(f"No descriptions available for {table_name}")
            else:
                st.warning("No column descriptions found in database")
                
                # Show table summary
                st.subheader("📊 Available Tables")
                schema_info = db_manager.get_table_info(db_schema)
                if schema_info:
                    table_summary = []
                    for table_name, columns in schema_info.items():
                        table_summary.append({
                            'Table': table_name,
                            'Columns': len(columns),
                            'Description': f"Table with {len(columns)} columns"
                        })
                    
                    summary_df = pd.DataFrame(table_summary)
                    st.dataframe(summary_df, use_container_width=True)
        else:
            # Main Text-to-SQL page
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.header("Natural Language Query")
                query = st.text_area(
                    "Enter your query:",
                    placeholder="e.g., Find all customers from New York with orders over $1000",
                    height=120
                )
                
                if st.button("Generate SQL", type="primary"):
                    if query.strip():
                        with st.spinner("🤖 Generating SQL..."):
                            llm = initialize_openai()
                            if llm:
                                # Build schema
                                schema_info = db_manager.get_table_info(db_schema)
                                schema_text = ""
                                for table_name, columns in schema_info.items():
                                    schema_text += f"\nTable: {table_name}\n"
                                    for column in columns[:8]:  # Limit columns
                                        pk = " (PK)" if column['primary_key'] else ""
                                        schema_text += f"  - {column['name']}: {column['type']}{pk}\n"
                                    if len(columns) > 8:
                                        schema_text += f"  ... and {len(columns) - 8} more\n"
                                
                                # Build sample data
                                sample_data_text = ""
                                for table_name in list(schema_info.keys())[:2]:  # Limit tables
                                    sample_data = db_manager.get_sample_data(table_name, 1, db_schema)
                                    if sample_data:
                                        sample_data_text += f"\n{table_name}: {sample_data[0]}\n"
                                
                                # Get foreign key relations for the prompt
                                relations = db_manager.get_foreign_key_relations(db_schema)
                                
                                generated_sql = generate_sql(llm, schema_text, sample_data_text, query, relations, db_manager, db_schema)
                                st.session_state.generated_sql = generated_sql
                    else:
                        st.error("Please enter a query")
            
            with col2:
                st.header("Generated SQL")
                if 'generated_sql' in st.session_state:
                    st.code(st.session_state.generated_sql, language="sql")
                    
                    if st.button("Execute Query"):
                        try:
                            result = db_manager.execute_query(st.session_state.generated_sql)
                            st.session_state.query_result = result
                            st.success("✅ Query executed!")
                        except Exception as e:
                            st.error(f"❌ Execution failed: {str(e)}")
            
            # Results
            if 'query_result' in st.session_state:
                st.header("Query Results")
                if st.session_state.query_result:
                    df = pd.DataFrame(st.session_state.query_result)
                    st.dataframe(df, use_container_width=True)
                    st.info(f"📊 {len(df)} rows returned")
                else:
                    st.info("No results returned")
    
    else:
        st.info("🔗 Connect to PostgreSQL using the sidebar to get started")
        st.markdown("""
        ### Features:
        - 🔍 **Dynamic PostgreSQL**: Connect to any PostgreSQL database
        - 🤖 **GPT-4o-mini**: AI-powered natural language to SQL
        - 📊 **Schema Analysis**: Automatic table and column detection
        - ⚡ **Optimized**: Token-efficient for large databases
        
        ### Requirements:
        - PostgreSQL database
        - OpenAI API key in config.env
        - Valid database credentials
        """)

if __name__ == "__main__":
    main() 