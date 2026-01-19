from main import build_prompt

def test_mapping():
    row = {
        'source_column': 'customer_id',
        'source_sub_column': None,
        'transformation_logic': '',
        'target_column': 'cust_id',
        'target_sub_column': None,
        'source_data_type': 'string',
        'source_sub_data_type': None,
        'target_data_type': 'string',
        'target_sub_data_type': None
    }

    result = build_prompt(row)

    assert result.strip() == "customer_id AS cust_id"
