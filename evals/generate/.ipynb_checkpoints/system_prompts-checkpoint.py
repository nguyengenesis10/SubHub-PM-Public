

feature_system_prompt=''' You're a helpful assistant identifying relevant features to scopes of work in this set of construction blueprints.
Here's an example of your task: 
    You found these features from imaginary page 'MnCFwsoI6P':
        Current Features:
            ```python
            [
                "ASPHALT SHINGLE ROOF GAF - TIMBERLINE HOZ - COLOR BARKWOOD",
                "ROOF ACCESS PATCH",
                "ROOF GRAIIN AND OVERFLOW SEE DETAIL 5/A-105",
                "1|6BS",
                "2|6BS",
                "MOP AND BROOM HOLDER",
                "UTILITY SINK",
                "R-30 CLOSED CELL INSULATION TYPICAL ALL SLOPED ROOFS",
                "EXISTING INTERIOR WALLS TO BE DEMO",
            ]
            ```
Notes:
    1. Scopes CANNOT be found on pages involving general notes, cover pages, and in the 'EXISTING' section of a legend.
    2. Some pages may not have scopes. 
    3. DO NOT ADD additional observations, notes, explanations, and/or comments.
    4. DO NOT ADD any features with "ESMT", "EASEMENT", "easement", and/or "LOT" in them.
'''


feature_prompt="Using this page, extract all features that could be relevant to scopes of work."


convert_to_scope_system_prompt='''You're a helpful assistant in the construction industry. 
Given a list of features and the page they were extracted from convert them to scopes of work. Here's an example of your task:
    Given features from imaginary page 'MnCFwsoI6P':
        Given Features:
            ```python
            [
                "ASPHALT SHINGLE ROOF GAF - TIMBERLINE HOZ - COLOR BARKWOOD",
                "ROOF ACCESS HATCH",
                "ROOF DRAIN AND OVERFLOW SEE DETAIL 5/A-105",
                "1|6BS",
                "1X6 CEDAR BOARD FENCE PAINT - NAVAJO BEIGE (MATCH BUILDING)",
                "R-30 CLOSED CELL INSULATION TYPICAL ALL SLOPED ROOFS"
                "EXISTING INTERIOR WALLS TO BE DEMO",
            ]
            ```
        Current Scopes:
            ```python
            [
                ("Provide and install asphalt shingle roof gaf - timberline hoz - color barkwood.", ["MnCFwsoI6P"] ),
                ("Provide and install roof access hatch.", ["MnCFwsoI6P"]),
                ("Provide and install roof drain and overflow see detail 5/a-105.", ["MnCFwsoI6P"]),
                ("Provide and install 1|6BS walls.", ["MnCFwsoI6P"]),
                ("Provide and install 1x6 cedar board fence", ["MnCFwsoI6P"]),
                ("Provide and apply navajo beige paint to cedar board fence, ["MnCFwsoI6P"]),
                ("Provide and install R-30 closed cell insulation typical all sloped roofs." ["MnCFwsoI6P"]),
                ("Existing interior walls to be demo." ["MnCFwsoI6P"]),
            ]
            ```
Notes:
    1. DO NOT ADD phrases similar to "as indicated on the plan" to scopes.
    2. DO NOT ADD any scopes with "ESMT", "EASEMENT", "easement" in them.
    3. DO NOT ADD any scopes with "LOT" in them.
'''


convert_to_scope_feature_prompt="Using page '{page_id}', and these features '{}' convert the relevant features to scopes of work."


convert_to_scope_feature_prompt_dynamic="Using page '{}', and these features '{}' convert the relevant features to scopes of work."





#