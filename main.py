import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import re
import os
from langchain_core.runnables import RunnableLambda

# input and output files
INPUT_FILE = os.getenv("INPUT_FILE", "transformation_files/Transformation_logic.xlsx")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "transformation_files/transformed_with_sql.xlsx")

def running_in_docker() -> bool:
    return os.path.exists("/.dockerenv")

OLLAMA_BASE_URL = (
    "http://host.docker.internal:11434"
    if running_in_docker()
    else "http://localhost:11434"
)

LLM_HOST = "host.docker.internal" if running_in_docker() else "localhost"
llm = ChatOllama(model="gemma3:4b", base_url=OLLAMA_BASE_URL, temperature=0)


# building prompts as per requirement 
def build_prompt(row, multi_fields=None, exploded_alias=None):
    """
    exploded_alias: str or None. If source array exploded but target is scalar.
    multiple_fields: list of dict rows for struct/array combination
    """
    if multi_fields:
        source_cols = ", ".join(f"{r['source_column']}" for r in multi_fields)
        sub_cols = ", ".join(f"{r['source_sub_column']}" for r in multi_fields)
        transformation_logic = "; ".join(
            f"{r['transformation_logic']} AS {r['target_sub_column']}" if pd.notna(r['transformation_logic']) and str(r['transformation_logic']).strip() != "" 
            else f"{r['source_column']} AS {r['target_sub_column']}"
            for r in multi_fields
        )
        prompt_text = f"""
        You are a senior data engineer.
        Generate ONLY the SQL expression (not a full query) for the following array target.

        Combine the following source columns into a single array/struct:
        - source columns: {source_cols}
        - sub-columns: {sub_cols}
        - transformation logic: {transformation_logic}

        Rules:
        - Use only Apache Spark SQL / Databricks SQL functions
        - String, numeric, and boolean constants MUST be written as SQL literals
          (e.g. 'ABC', 123, true). **NEVER use lit()**
        - For direct mapping (no transformation), always use: source_sub_column AS target_column. Do NOT swap or modify these names.
        - Use struct() to combine multiple sub-columns
        - If target data type is array, wrap the struct() in ARRAY(...), e.g. ARRAY(STRUCT(...))
        - If the target data type and source data type are both scalar, strictly do NOT wrap the expression in STRUCT() or ARRAY(). Just generate the transformation logic
        - Direct mapping from source column or source sub column to target if transformation is empty. Strictly follow this example pattern: source column as target column. For example: customer_id as cust_id should remain same
        - Use CASE WHEN only if conditional logic exists
        - If any source column is an array, use the exploded alias {exploded_alias} in the expression
        - Strictly DO NOT use SELECT, FROM, EXPLODE(), ARRAY_ELEMENT_AT(), lit, or any other non-Spark SQL functions
        - Alias output as the target column only using AS
        - Return EXACTLY the expression, nothing else, no extra text, and strictly no SQL clause like SELECT or FROM table name
        - Strictly no SQL clause like SELECT or FROM table name or any other SQL clause inside ARRAY(....)
        """
    else:
        source_column_ref = row['source_column']
        # Using exploded_alias if source is an array and target is scalar
        if exploded_alias:
            source_column_ref = f"{exploded_alias}.{row['source_sub_column'] or row['source_column']}"

        # If transformation_logic is empty, direct mapping from source to target
        if pd.isna(row['transformation_logic']) or str(row['transformation_logic']).strip() == "":
            return f"{source_column_ref} AS {row['target_sub_column'] or row['target_column']}"

        prompt_text = f"""
        You are a senior data engineer.
        Generate ONLY the SQL expression (not a full query) for this field.

        Source:
        - column: {source_column_ref}
        - sub column: {row['source_sub_column']}
        - data type: {row['source_data_type']}
        - sub data type: {row['source_sub_data_type']}

        Transformation logic:
        {row['transformation_logic']}

        Target:
        - column: {row['target_column']}
        - sub column: {row['target_sub_column']}
        - data type: {row['target_data_type']}
        - sub data type: {row['target_sub_data_type']}

        Rules:
        - Use only Apache Spark SQL / Databricks SQL functions
        - String, numeric, and boolean constants MUST be written as SQL literals
          (e.g. 'ABC', 123, true). **NEVER use lit()**
        - Never use aggregate functions like sum unless explicitly mentioned so
        - Never wrap in struct() or in array() if target data type is not array
        - If the target data type is SCALAR like string, int, etc, strictly alias the output using AS {row['target_column']}
        - Use struct() if combining multiple sub-columns and only if target data type is array
        - Only if target data type is array, wrap the struct() in ARRAY(...), e.g. ARRAY(STRUCT(...))
        - If the target data type and source data type are both scalar, strictly do NOT wrap the expression in STRUCT() or ARRAY(). Just generate the transformation logic and alias it as the target column
        - Direct mapping from source column or source sub column to target if transformation is empty
        - Use CASE WHEN only if conditional logic exists
        - Use exploded alias {exploded_alias} if source is array but target is scalar
        - Alias output only with AS
        - Return EXACTLY the expression, nothing else, no extra text, and strictly no SQL clause like SELECT or FROM table name
        - Strictly no SQL clause like SELECT or FROM table name or any other SQL clause inside ARRAY(....)
        """
    return prompt_text

def normalize_sql(sql: str) -> str:
    if pd.isna(sql) or sql is None:
        return ""
    sql = sql.strip().lower()   #removes all leading and trailing spaces
    sql = sql.replace("`", "")  #removes backticks
    sql = re.sub(r"\s+", " ", sql) #removes sequential white space charcters
    sql = sql.rstrip(";") #removes semi colon
    sql = re.sub(r"\s*([=<>(),])\s*", r"\1", sql) #removes spaces around symbols
    return sql

def main():
    df = pd.read_excel(INPUT_FILE).copy()

    # Add columns to store LLM result and exact match
    df['generated_sql_expression'] = ""
    df['lateral_exploded_alias'] = ""
    df['exact_match'] = ""
    df['match_test_using_LLM'] = ""
    df['improved_generated_sql_expression'] = ""

    array_fields = []
    for idx, row in df.iterrows():
        if row['source_data_type'] == 'array':
            exploded_alias_name = f"{row['source_column'][:3]}_{row['source_column'][-3:]}_expl"
            sub_col = row['source_sub_column']
            target_table = row['target_table']
            source_column = row['source_column']
            array_fields.append((exploded_alias_name, sub_col, target_table, source_column))

    grouped = df.groupby(['target_table', 'target_column'])

    #created two chains for two types of data
    scalar_chain = (
        RunnableLambda(lambda x: {"prompt": build_prompt(x["row"], exploded_alias=x.get("exploded_alias")), **x}) |
        RunnableLambda(lambda x: {"uncleanedsql": llm.invoke([HumanMessage(content=x["prompt"])]).content, **x}) |
        RunnableLambda(lambda x: {"generated_sql": x["uncleanedsql"].replace("```","").replace("sql","").replace("`","").strip(), **x}) |
        RunnableLambda(lambda x: {
            "exact_match": normalize_sql(x["generated_sql"]) == normalize_sql(x["expected_sql"]),**x
        }) |
        RunnableLambda(lambda x: {
            "match_LLM": llm.invoke([HumanMessage(content=f"Compare two SQL statemtents and generate only true or false depending on whether they are equivalent or not. Two sqls are: {x["generated_sql"]} and {x["expected_sql"]}")]).content.strip().lower() == "true", **x
        }) |
        RunnableLambda(lambda x: {
            "new_prompt": f""" For transformation logic: {x["transformation_logic"]} \n Generated sql: \n{x["generated_sql"]} is not generated properly. \n Generated sql should have been like this:\n {x["expected_sql"]}. Take this expecteed example as reference and make sure these kind of issues won't reappear in future.
            """ if not x["match_LLM"] else '', **x
        }) |
        RunnableLambda(lambda x: {"newuncleanedsql": llm.invoke([HumanMessage(content=x["prompt"]+ "\n\n" + x["new_prompt"])]).content.strip(), **x
        }) | 
        RunnableLambda(lambda x: {"improved_generated_sql_expression": x["newuncleanedsql"].replace("```","").replace("sql","").replace("`","").strip() if not x["match_LLM"] else '', **x})
    )
    array_chain = (
        RunnableLambda(lambda x: {"prompt": build_prompt(None, multi_fields=x["multi_fields"]), **x}) |
        RunnableLambda(lambda x: {"uncleanedsql": llm.invoke([HumanMessage(content=x["prompt"])]).content, **x}) |
        RunnableLambda(lambda x: {"generated_sql": x["uncleanedsql"].replace("```","").replace("sql","").replace("`","").strip(), **x}) |
        RunnableLambda(lambda x: {
            "exact_match": 
                normalize_sql(x["generated_sql"]) == normalize_sql(x["expected_sql"]), **x
        }) |
        RunnableLambda(lambda x: {
            "match_LLM": llm.invoke([HumanMessage(content=f"Compare two SQL statemtents and generate only true or false depending on whether they are equivalent or not. Two sqls are: {x["generated_sql"]} and {x["expected_sql"]}")]).content.strip().lower() == "true", **x
        }) |
        RunnableLambda(lambda x: {
            "new_prompt": f""" For transformation logic: {x["transformation_logic"]} \n Generated sql: \n{x["generated_sql"]} is not generated properly. \n Generated sql should have been like this:\n {x["expected_sql"]}. Take this expecteed example as reference and make sure these kind of issues won't reappear in future.
            """ if not x["match_LLM"] else '', **x
        }) |
        RunnableLambda(lambda x: {"newuncleanedsql": llm.invoke([HumanMessage(content=x["prompt"]+ "\n\n" + x["new_prompt"])]).content.strip(), **x
        }) | 
        RunnableLambda(lambda x: {"improved_generated_sql_expression": x["newuncleanedsql"].replace("```","").replace("sql","").replace("`","").strip() if not x["match_LLM"] else '', **x})
    )

    #processing each group
    for (_, _), group in grouped:
        # target with array data type
        if group['target_data_type'].iloc[0] == 'array' and len(group) > 1:
            expected_sql = group['Expected_SQL'].iloc[0] if 'Expected_SQL' in group.columns else ""
            result = array_chain.invoke({
                "multi_fields": group.to_dict('records'),
                "expected_sql": expected_sql,
            })
            df.loc[group.index[0], 'generated_sql_expression'] = result["generated_sql"]
            df.loc[group.index[0], 'exact_match'] = result["exact_match"]
            df.loc[group.index[0], 'match_test_using_LLM'] = result["match_LLM"]
            df.loc[group.index[0], 'improved_generated_sql_expression'] = result["improved_generated_sql_expression"]
            continue

        # target with scalar datatype
        for idx, row in group.iterrows():
            if pd.isna(row['target_column']):
                continue

            exploded_alias_name = None
            if row['source_data_type'] == 'array' and row['target_data_type'] != 'array':
                exploded_alias_name = f"{row['source_column'][:3]}_{row['source_column'][-3:]}_expl"
                df.at[idx, 'lateral_exploded_alias'] = f"LATERAL VIEW EXPLODE({row['source_column']}) AS {exploded_alias_name}"

                if pd.notna(row['transformation_logic']) and str(row['transformation_logic']).strip() != "":
                    for alias, sub_col, tgt_table, srccolumn in array_fields:
                        if tgt_table == row['target_table'] and srccolumn == row['source_column'] and sub_col:
                            row['transformation_logic'] = re.sub(
                                rf'\b{sub_col}\b', f"{alias}.{sub_col}", row['transformation_logic']
                            )

            expected_sql = row.get("Expected_SQL", "")

            # Direct mapping
            if pd.isna(row['transformation_logic']) or str(row['transformation_logic']).strip() == "":
                source_ref = (exploded_alias_name + "." + (row['source_sub_column'] or row['source_column'])
                              if exploded_alias_name else row['source_column'])
                df.at[idx, 'generated_sql_expression'] = f"{source_ref} AS {row['target_column']}"
                df.at[idx, 'exact_match'] = normalize_sql(f"{source_ref} AS {row['target_column']}") == normalize_sql(expected_sql)
                df.at[idx, 'match_test_using_LLM']= normalize_sql(f"{source_ref} AS {row['target_column']}") == normalize_sql(expected_sql) # this is not LLM generated but still used as for direct mapping LLM won't be necessary; but if previous one is not done then similar logic to else condition can be used
            else:
                # Use scalar_chain
                result = scalar_chain.invoke({
                    "row": row,
                    "exploded_alias": exploded_alias_name,
                    "expected_sql": expected_sql,
                    "transformation_logic": row['transformation_logic']
                })
                df.at[idx, 'generated_sql_expression'] = result["generated_sql"]
                df.at[idx, 'exact_match'] = result["exact_match"]
                df.at[idx, 'match_test_using_LLM'] = result["match_LLM"]
                df.at[idx, 'improved_generated_sql_expression'] = result["improved_generated_sql_expression"]

    
    #saving output in the above mentioned excel file
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"SQL expressions written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()